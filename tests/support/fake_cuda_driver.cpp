#include "fake_cuda_driver.hpp"

#include "vramz/checked.hpp"

#include <algorithm>
#include <bit>
#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace vramz::test {
using detail::CudaAllocationProperties;
using detail::CudaAttribute;
using detail::CudaContext;
using detail::CudaDevice;
using detail::CudaPhysicalHandle;
namespace {
[[nodiscard]] Error failure(CudaCall call, bool ambiguous = false) noexcept {
    const auto code = ambiguous ? ErrorCode::ambiguous_backend_state
                                : (call == CudaCall::create || call == CudaCall::reserve
                                       ? ErrorCode::out_of_gpu_memory
                                       : ErrorCode::backend_failure);
    auto operation = OperationId::allocate;
    if (call == CudaCall::completion_create) {
        operation = OperationId::defer_lease;
    } else if (call == CudaCall::completion_query) {
        operation = OperationId::query_completion;
    } else if (call == CudaCall::completion_release) {
        operation = OperationId::release_completion;
    }
    return Error{code,
                 operation,
                 BackendId{2U},
                 NativeErrorDomain::internal_backend,
                 static_cast<std::int64_t>(call),
                 0U,
                 0U};
}
[[nodiscard]] bool mode(const std::optional<CudaFault>& fault, CudaFaultMode expected) noexcept {
    return fault && fault->mode == expected;
}
[[nodiscard]] Error invalid() noexcept {
    return make_error(ErrorCode::invalid_argument, OperationId::allocate);
}
} // namespace

FakeCudaDriverApi::FakeCudaDriverApi(std::shared_ptr<FakeCudaState> state) noexcept
    : state_(std::move(state)) {
    if (!state_) {
        std::terminate();
    }
}
FakeCudaDriverApi::~FakeCudaDriverApi() {
    const std::scoped_lock lock{state_->mutex};
    ++state_->destroyed_drivers;
}
void FakeCudaDriverApi::inject(CudaCall point, CudaFaultMode fault_mode,
                               CudaFaultOptions options) noexcept {
    const std::scoped_lock lock{state_->mutex};
    if (point == CudaCall::count || options.after_calls == 0U) {
        std::terminate();
    }
    const auto occurrence = checked_add(state_->counts[static_cast<std::size_t>(point)],
                                        options.after_calls, OperationId::verify);
    auto slot = std::ranges::find_if(
        state_->faults, [](const auto& fault) { return fault.call == CudaCall::count; });
    if (!occurrence || slot == state_->faults.end()) {
        std::terminate();
    }
    *slot = CudaFault{point, occurrence.value(), fault_mode, options.detail};
}
std::optional<CudaFault> FakeCudaDriverApi::call(CudaCall point, std::uint64_t identity,
                                                 ByteSize size) noexcept {
    auto& count = state_->counts[static_cast<std::size_t>(point)];
    if (count == std::numeric_limits<std::uint64_t>::max()) {
        std::terminate();
    }
    ++count;
    if (state_->log_size < state_->log.size()) {
        state_->log[state_->log_size++] = CudaCallRecord{point, identity, size};
    } else if (state_->dropped_calls != std::numeric_limits<std::uint64_t>::max()) {
        ++state_->dropped_calls;
    }
    for (auto& fault : state_->faults) {
        if (fault.call == point && fault.occurrence == count) {
            const auto result = fault;
            fault = {};
            return result;
        }
    }
    return {};
}
std::uint64_t FakeCudaDriverApi::count(CudaCall point) const noexcept {
    const std::scoped_lock lock{state_->mutex};
    return point == CudaCall::count ? 0U : state_->counts[static_cast<std::size_t>(point)];
}
bool FakeCudaDriverApi::empty() const noexcept {
    const std::scoped_lock lock{state_->mutex};
    return state_->owned_bytes == 0U && state_->primary_references == 0U &&
           std::ranges::none_of(
               state_->completions,
               [](const auto& entry) { return entry.id != CompletionTokenId{}; }) &&
           std::ranges::none_of(
               state_->reservations,
               [](const auto& entry) { return entry.address != DeviceAddress{}; }) &&
           std::ranges::none_of(
               state_->mappings,
               [](const auto& entry) { return entry.address != DeviceAddress{}; }) &&
           std::ranges::none_of(state_->contexts,
                                [](const auto& entry) { return entry.depth != 0U; });
}
FakeCudaState::ThreadContext* FakeCudaDriverApi::thread_context() noexcept {
    const auto current = std::this_thread::get_id();
    auto found = std::ranges::find_if(
        state_->contexts, [current](const auto& entry) { return entry.thread == current; });
    if (found != state_->contexts.end()) {
        return &*found;
    }
    found =
        std::ranges::find_if(state_->contexts, [](const auto& entry) { return entry.depth == 0U; });
    if (found == state_->contexts.end()) {
        return nullptr;
    }
    found->thread = current;
    return &*found;
}
bool FakeCudaDriverApi::valid_context() noexcept {
    const auto* context = thread_context();
    return state_->initialized && context != nullptr && context->depth != 0U &&
           context->stack[context->depth - 1U] == CudaContext{1U} &&
           state_->primary_references != 0U;
}
CudaContext FakeCudaDriverApi::current_context() const noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto current = std::this_thread::get_id();
    const auto found = std::ranges::find_if(
        state_->contexts, [current](const auto& entry) { return entry.thread == current; });
    return found != state_->contexts.end() && found->depth != 0U ? found->stack[found->depth - 1U]
                                                                 : CudaContext{};
}
Result<bool> FakeCudaDriverApi::context_is_current(CudaContext context) noexcept {
    return context != CudaContext{} && context == current_context();
}

Result<void> FakeCudaDriverApi::copy_device_to_device(DeviceAddress destination,
                                                      DeviceAddress source,
                                                      ByteSize size) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::device_to_device, destination.value(), size);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::device_to_device);
    }
    auto* from_map = mapping(source);
    auto* to_map = mapping(destination);
    if (!valid_context() || from_map == nullptr || to_map == nullptr || !from_map->read_write ||
        !to_map->read_write ||
        size.value() > from_map->size.value() - (source.value() - from_map->address.value()) ||
        size.value() > to_map->size.value() - (destination.value() - to_map->address.value())) {
        return invalid();
    }
    auto* from = physical(from_map->handle);
    auto* to = physical(to_map->handle);
    if (from == nullptr || to == nullptr || from == to) {
        return invalid();
    }
    const auto input = std::span{from->bytes}.subspan(
        static_cast<std::size_t>(source.value() - from_map->address.value()),
        static_cast<std::size_t>(size.value()));
    auto output = std::span{to->bytes}.subspan(
        static_cast<std::size_t>(destination.value() - to_map->address.value()), input.size());
    std::ranges::copy(input, output.begin());
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::device_to_device, true);
    }
    if (mode(fault, CudaFaultMode::malformed) && !output.empty()) {
        output.front() ^= std::byte{1U};
    }
    return {};
}

FakeCudaState::Physical* FakeCudaDriverApi::physical(CudaPhysicalHandle handle) noexcept {
    const auto found = std::ranges::find_if(state_->physical, [handle](const auto& entry) {
        return handle != CudaPhysicalHandle{} && entry.handle == handle;
    });
    return found == state_->physical.end() ? nullptr : &*found;
}
FakeCudaState::Mapping* FakeCudaDriverApi::mapping(DeviceAddress address) noexcept {
    const auto found = std::ranges::find_if(state_->mappings, [address](const auto& entry) {
        return entry.address != DeviceAddress{} && address >= entry.address &&
               address.value() - entry.address.value() < entry.size.value();
    });
    return found == state_->mappings.end() ? nullptr : &*found;
}
Result<void> FakeCudaDriverApi::initialize() noexcept {
    const std::scoped_lock lock{state_->mutex};
    if (call(CudaCall::initialize)) {
        return failure(CudaCall::initialize);
    }
    if (state_->config.initialize_native_error != 0) {
        return Error{ErrorCode::backend_failure,
                     OperationId::runtime_create,
                     BackendId{2U},
                     NativeErrorDomain::cuda_driver,
                     state_->config.initialize_native_error,
                     0U,
                     0U};
    }
    state_->initialized = true;
    return {};
}
Result<std::int32_t> FakeCudaDriverApi::version() noexcept {
    const std::scoped_lock lock{state_->mutex};
    if (call(CudaCall::version)) {
        return failure(CudaCall::version);
    }
    if (!state_->initialized) {
        return invalid();
    }
    return state_->config.version;
}
Result<CudaDevice> FakeCudaDriverApi::device(std::int32_t ordinal) noexcept {
    const std::scoped_lock lock{state_->mutex};
    if (call(CudaCall::device)) {
        return failure(CudaCall::device);
    }
    if (!state_->initialized || ordinal != 0) {
        return invalid();
    }
    return CudaDevice{0};
}
Result<detail::CudaDeviceInfo> FakeCudaDriverApi::device_info(CudaDevice selected) noexcept {
    const std::scoped_lock lock{state_->mutex};
    if (call(CudaCall::device_info)) {
        return failure(CudaCall::device_info);
    }
    if (!state_->initialized || selected != 0) {
        return invalid();
    }
    auto info = state_->config.device_info;
    info.name.back() = '\0';
    return info;
}
Result<bool> FakeCudaDriverApi::attribute(CudaDevice selected,
                                          CudaAttribute attribute_id) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto point =
        attribute_id == CudaAttribute::unified_addressing ? CudaCall::uva : CudaCall::vmm;
    if (call(point)) {
        return failure(point);
    }
    if (!state_->initialized || selected != 0) {
        return invalid();
    }
    return attribute_id == CudaAttribute::unified_addressing ? state_->config.unified_addressing
                                                             : state_->config.vmm;
}
Result<ByteSize> FakeCudaDriverApi::host_page_size() noexcept {
    const std::scoped_lock lock{state_->mutex};
    if (call(CudaCall::host_page)) {
        return failure(CudaCall::host_page);
    }
    return state_->config.host_page;
}
Result<ByteSize> FakeCudaDriverApi::granularity(CudaDevice selected, bool recommended) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto point = recommended ? CudaCall::recommended : CudaCall::minimum;
    if (call(point)) {
        return failure(point);
    }
    if (!state_->initialized || selected != 0) {
        return invalid();
    }
    return recommended ? state_->config.recommended : state_->config.minimum;
}
Result<CudaContext> FakeCudaDriverApi::retain_primary(CudaDevice selected) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::retain_primary);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::retain_primary);
    }
    if (!state_->initialized || selected != 0 ||
        state_->primary_references == std::numeric_limits<std::uint32_t>::max()) {
        return invalid();
    }
    ++state_->primary_references;
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::retain_primary, true);
    }
    return CudaContext{mode(fault, CudaFaultMode::malformed) ? 0U : 1U};
}
Result<void> FakeCudaDriverApi::release_primary(CudaDevice selected) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::release_primary);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::release_primary);
    }
    if (selected != 0 || state_->primary_references == 0U || state_->owned_bytes != 0U ||
        std::ranges::any_of(state_->completions,
                            [](const auto& entry) { return entry.id != CompletionTokenId{}; }) ||
        std::ranges::any_of(state_->contexts, [](const auto& entry) {
            return entry.depth != 0U && entry.stack[entry.depth - 1U] == CudaContext{1U};
        })) {
        return invalid();
    }
    --state_->primary_references;
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::release_primary, true);
    }
    return {};
}
Result<void> FakeCudaDriverApi::push_context(CudaContext context) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::push, context.value());
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::push);
    }
    auto* stack = thread_context();
    if (!state_->initialized || stack == nullptr || stack->depth == stack->stack.size() ||
        context == CudaContext{}) {
        return invalid();
    }
    stack->stack[stack->depth++] =
        mode(fault, CudaFaultMode::malformed) ? CudaContext{77U} : context;
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::push, true);
    }
    return {};
}
Result<CudaContext> FakeCudaDriverApi::pop_context() noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::pop);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::pop);
    }
    auto* stack = thread_context();
    if (stack == nullptr || stack->depth == 0U) {
        return invalid();
    }
    const auto popped = stack->stack[--stack->depth];
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::pop, true);
    }
    return mode(fault, CudaFaultMode::malformed) ? CudaContext{77U} : popped;
}
Result<DeviceAddress> FakeCudaDriverApi::reserve(detail::CudaAddressRequest request) noexcept {
    const auto size = request.size;
    const auto alignment = request.alignment;
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::reserve, 0U, size);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::reserve);
    }
    if (!valid_context() || size.value() == 0U || state_->config.host_page.value() == 0U ||
        size.value() % state_->config.host_page.value() != 0U ||
        !std::has_single_bit(alignment.value())) {
        return invalid();
    }
    auto slot = std::ranges::find_if(
        state_->reservations, [](const auto& entry) { return entry.address == DeviceAddress{}; });
    const auto base =
        checked_align_up(state_->next_address, alignment.value(), OperationId::allocate);
    if (!base || slot == state_->reservations.end()) {
        return failure(CudaCall::reserve);
    }
    const auto end = checked_add(base.value(), size.value(), OperationId::allocate);
    if (!end) {
        return end.error();
    }
    *slot = FakeCudaState::Reservation{DeviceAddress{base.value()}, size};
    state_->next_address = end.value();
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::reserve, true);
    }
    if (fault && fault->mode == CudaFaultMode::malformed) {
        const auto displaced = checked_add(
            base.value(), fault->detail == 0U ? 1U : alignment.value(), OperationId::allocate);
        if (!displaced) {
            return failure(CudaCall::reserve, true);
        }
        return DeviceAddress{displaced.value()};
    }
    return slot->address;
}
Result<void> FakeCudaDriverApi::free_address(DeviceAddress address, ByteSize size) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::free_address, address.value(), size);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::free_address);
    }
    const auto slot =
        std::ranges::find_if(state_->reservations, [address, size](const auto& entry) {
            return entry.address == address && entry.size == size;
        });
    if (!valid_context() || address == DeviceAddress{} || slot == state_->reservations.end() ||
        std::ranges::any_of(state_->mappings, [address, size](const auto& entry) {
            return entry.address >= address &&
                   entry.address.value() - address.value() < size.value();
        })) {
        return invalid();
    }
    *slot = {};
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::free_address, true);
    }
    return {};
}
Result<CudaPhysicalHandle> FakeCudaDriverApi::create(ByteSize size, CudaDevice selected) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::create, 0U, size);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::create);
    }
    if (!valid_context() || selected != 0 || size.value() == 0U ||
        state_->config.minimum.value() == 0U ||
        size.value() % state_->config.minimum.value() != 0U) {
        return invalid();
    }
    auto slot = std::ranges::find_if(
        state_->physical, [](const auto& entry) { return entry.handle == CudaPhysicalHandle{}; });
    const auto total = checked_add(state_->owned_bytes, size.value(), OperationId::allocate);
    if (slot == state_->physical.end() || !total ||
        total.value() > state_->config.physical_limit.value() ||
        size > state_->config.maximum_resource || size.value() > 32ULL * 1024ULL * 1024ULL ||
        size.value() > slot->bytes.max_size() ||
        state_->next_handle == std::numeric_limits<std::uint64_t>::max()) {
        return failure(CudaCall::create);
    }
    try {
        slot->bytes.resize(static_cast<std::size_t>(size.value()));
    } catch (const std::bad_alloc&) {
        return make_error(ErrorCode::out_of_host_memory, OperationId::allocate);
    }
    slot->handle = CudaPhysicalHandle{state_->next_handle++};
    slot->references = 1U;
    slot->properties = CudaAllocationProperties{selected, true, true, true, true};
    state_->owned_bytes = total.value();
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::create, true);
    }
    return mode(fault, CudaFaultMode::malformed) ? CudaPhysicalHandle{} : slot->handle;
}
Result<CudaAllocationProperties> FakeCudaDriverApi::properties(CudaPhysicalHandle handle) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::properties, handle.value());
    if (mode(fault, CudaFaultMode::before) || mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::properties);
    }
    const auto* resource = physical(handle);
    if (!valid_context() || resource == nullptr) {
        return invalid();
    }
    auto result = resource->properties;
    if (fault && fault->mode == CudaFaultMode::malformed) {
        switch (fault->detail) {
        case 0U:
            result.device = 1;
            break;
        case 1U:
            result.pinned = false;
            break;
        case 2U:
            result.no_export = false;
            break;
        case 3U:
            result.compression_disabled = false;
            break;
        default:
            result.device_local = false;
            break;
        }
    }
    return result;
}
Result<void> FakeCudaDriverApi::map(DeviceAddress address, ByteSize size,
                                    CudaPhysicalHandle handle) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::map, address.value(), size);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::map);
    }
    const auto* resource = physical(handle);
    const bool contained =
        std::ranges::any_of(state_->reservations, [address, size](const auto& entry) {
            return entry.address != DeviceAddress{} && address >= entry.address &&
                   address.value() - entry.address.value() <= entry.size.value() &&
                   size.value() <= entry.size.value() - (address.value() - entry.address.value());
        });
    auto slot = std::ranges::find_if(
        state_->mappings, [](const auto& entry) { return entry.address == DeviceAddress{}; });
    if (!valid_context() || resource == nullptr || resource->bytes.size() != size.value() ||
        !contained || slot == state_->mappings.end() || state_->config.minimum.value() == 0U ||
        address.value() % state_->config.minimum.value() != 0U) {
        return invalid();
    }
    for (const auto& existing : state_->mappings) {
        if (existing.address == DeviceAddress{}) {
            continue;
        }
        if (address >= existing.address
                ? address.value() - existing.address.value() < existing.size.value()
                : existing.address.value() - address.value() < size.value()) {
            return invalid();
        }
    }
    *slot = FakeCudaState::Mapping{address, size, handle, false};
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::map, true);
    }
    return {};
}
Result<void> FakeCudaDriverApi::set_access(DeviceAddress address, ByteSize size,
                                           CudaDevice selected) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::set_access, address.value(), size);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::set_access);
    }
    auto* entry = mapping(address);
    if (!valid_context() || selected != 0 || entry == nullptr || entry->address != address ||
        entry->size != size) {
        return invalid();
    }
    entry->read_write = !mode(fault, CudaFaultMode::malformed);
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::set_access, true);
    }
    return {};
}
Result<bool> FakeCudaDriverApi::has_read_write_access(DeviceAddress address,
                                                      CudaDevice selected) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::get_access, address.value());
    if (mode(fault, CudaFaultMode::before) || mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::get_access);
    }
    const auto* entry = mapping(address);
    if (!valid_context() || selected != 0 || entry == nullptr) {
        return invalid();
    }
    return mode(fault, CudaFaultMode::malformed) ? false : entry->read_write;
}
Result<CudaPhysicalHandle> FakeCudaDriverApi::retain_mapping(DeviceAddress address) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::retain_mapping, address.value());
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::retain_mapping);
    }
    const auto* entry = mapping(address);
    auto* resource = entry == nullptr ? nullptr : physical(entry->handle);
    if (!valid_context() || resource == nullptr ||
        resource->references == std::numeric_limits<std::uint32_t>::max()) {
        return invalid();
    }
    ++resource->references;
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::retain_mapping, true);
    }
    if (fault && fault->mode == CudaFaultMode::malformed) {
        return CudaPhysicalHandle{fault->detail == 0U ? 0U : 999U};
    }
    return resource->handle;
}
Result<void> FakeCudaDriverApi::unmap(DeviceAddress address, ByteSize size) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::unmap, address.value(), size);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::unmap);
    }
    auto* entry = mapping(address);
    if (!valid_context() || entry == nullptr || entry->address != address || entry->size != size) {
        return invalid();
    }
    *entry = {};
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::unmap, true);
    }
    return {};
}
Result<void> FakeCudaDriverApi::release(CudaPhysicalHandle handle) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::release, handle.value());
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::release);
    }
    auto* resource = physical(handle);
    if (!valid_context() || resource == nullptr || resource->references == 0U) {
        return invalid();
    }
    const bool mapped = std::ranges::any_of(
        state_->mappings, [handle](const auto& entry) { return entry.handle == handle; });
    // A validation reference can be released while mapped; the owner's final reference cannot.
    if (resource->references == 1U && mapped) {
        return invalid();
    }
    --resource->references;
    if (resource->references == 0U) {
        state_->owned_bytes -= resource->bytes.size();
        *resource = {};
    }
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::release, true);
    }
    return {};
}
Result<void> FakeCudaDriverApi::copy(DeviceAddress address, std::span<std::byte> destination,
                                     std::span<const std::byte> source, bool to_device) noexcept {
    const auto point = to_device ? CudaCall::host_to_device : CudaCall::device_to_host;
    const auto size = to_device ? source.size() : destination.size();
    const auto fault = call(point, address.value(), ByteSize{size});
    if (mode(fault, CudaFaultMode::before)) {
        return failure(point);
    }
    const auto* entry = mapping(address);
    auto* resource = entry == nullptr ? nullptr : physical(entry->handle);
    if (!valid_context() || entry == nullptr || !entry->read_write || resource == nullptr ||
        size > entry->size.value() - (address.value() - entry->address.value())) {
        return invalid();
    }
    auto bytes = std::span{resource->bytes}.subspan(
        static_cast<std::size_t>(address.value() - entry->address.value()), size);
    if (to_device) {
        std::ranges::copy(source, bytes.begin());
    } else {
        std::ranges::copy(bytes, destination.begin());
    }
    if (mode(fault, CudaFaultMode::after)) {
        return failure(point, to_device);
    }
    if (mode(fault, CudaFaultMode::malformed) && size != 0U) {
        if (to_device) {
            bytes.front() ^= std::byte{1};
        } else {
            destination.front() ^= std::byte{1};
        }
    }
    return {};
}
Result<void> FakeCudaDriverApi::copy_to_device(DeviceAddress destination,
                                               std::span<const std::byte> source) noexcept {
    const std::scoped_lock lock{state_->mutex};
    return copy(destination, {}, source, true);
}
Result<void> FakeCudaDriverApi::copy_from_device(std::span<std::byte> destination,
                                                 DeviceAddress source) noexcept {
    const std::scoped_lock lock{state_->mutex};
    return copy(source, destination, {}, false);
}

bool FakeCudaDriverApi::supports_external_completions() const noexcept {
    const std::scoped_lock lock{state_->mutex};
    return state_->config.external_completions;
}

Result<CompletionTokenId> FakeCudaDriverApi::create_completion() noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::completion_create);
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::completion_create);
    }
    if (!valid_context() || !state_->config.external_completions) {
        return invalid();
    }
    const auto slot = std::ranges::find_if(
        state_->completions, [](const auto& entry) { return entry.id == CompletionTokenId{}; });
    if (slot == state_->completions.end()) {
        return make_error(ErrorCode::busy, OperationId::defer_lease);
    }
    if (state_->next_completion_id == std::numeric_limits<std::uint64_t>::max()) {
        return make_error(ErrorCode::arithmetic_overflow, OperationId::defer_lease);
    }
    slot->id = CompletionTokenId{state_->next_completion_id++};
    slot->state = CompletionState::pending;
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::completion_create, true);
    }
    return mode(fault, CudaFaultMode::malformed) ? CompletionTokenId{} : slot->id;
}

Result<CompletionState> FakeCudaDriverApi::query_completion(CompletionTokenId id) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::completion_query, id.value());
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::completion_query);
    }
    const auto slot = std::ranges::find_if(state_->completions,
                                           [id](const auto& entry) { return entry.id == id; });
    if (!valid_context() || id == CompletionTokenId{} || slot == state_->completions.end()) {
        return invalid();
    }
    if (mode(fault, CudaFaultMode::after)) {
        return failure(CudaCall::completion_query, true);
    }
    return mode(fault, CudaFaultMode::malformed) ? CompletionState::ambiguous : slot->state;
}

Result<void> FakeCudaDriverApi::release_completion(CompletionTokenId id) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto fault = call(CudaCall::completion_release, id.value());
    if (mode(fault, CudaFaultMode::before)) {
        return failure(CudaCall::completion_release);
    }
    const auto slot = std::ranges::find_if(state_->completions,
                                           [id](const auto& entry) { return entry.id == id; });
    if (!valid_context() || id == CompletionTokenId{} || slot == state_->completions.end()) {
        return invalid();
    }
    if (slot->state != CompletionState::complete) {
        return make_error(ErrorCode::busy, OperationId::release_completion);
    }
    *slot = {};
    if (mode(fault, CudaFaultMode::after) || mode(fault, CudaFaultMode::malformed)) {
        return failure(CudaCall::completion_release, true);
    }
    return {};
}

Result<void> FakeCudaDriverApi::mark_completion_ready(CompletionTokenId id) noexcept {
    const std::scoped_lock lock{state_->mutex};
    const auto slot = std::ranges::find_if(state_->completions,
                                           [id](const auto& entry) { return entry.id == id; });
    if (id == CompletionTokenId{} || slot == state_->completions.end()) {
        return make_error(ErrorCode::stale_handle, OperationId::query_completion);
    }
    slot->state = CompletionState::complete;
    return {};
}

} // namespace vramz::test
