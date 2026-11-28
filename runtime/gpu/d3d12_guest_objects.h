#pragma once

// Guest-facing D3D12 objects for d3d12_xs.dll (G24), built on the guest COM bridge (O16,
// guest/com_bridge.h) and driving the G6 descriptor heaps / root signatures, the G7 PSO cache and
// the G20 state bridge.
//
// Guest code reaches every D3D12 method as `mov rax,[rcx]; call [rax+8*slot]` on interface pointers
// that d3d12_xs_exports.h hands out. Each interface below is a declarative com::declare_interface
// table in the slot order established from the real executable (static analysis, local only; only
// slot indices, argument counts, call-site counts and IIDs are recorded here):
//
// Evidence summary (game_scarlett_master.exe)
// -------------------------------------------
// * D3D12XboxCreateDevice is called once, with adapter = null, a 0x50-byte parameter block on the
//   stack and riid = IID_ID3D12Device11; the device lands in one global and is copied into 11 more.
//   Interface pointers were followed from there (globals, call arguments, return values, queue
//   wrapper objects): 42 distinct device slots in 54 functions (~93 call sites).
// * ID3D12Device: the public ID3D12Device order holds for slots 0..43. Confirmed by IID arguments
//   at the public creation slots (8 queue, 9 allocator, 14 descriptor heap, 16 root signature,
//   27 committed resource, 36 fence, 39 query heap, 41 command signature), by argument counts
//   (15 GetDescriptorHandleIncrementSize 2, 17 CreateConstantBufferView 3, 18 CreateShaderResourceView
//   4, 19 CreateUnorderedAccessView 5, 24 CopyDescriptorsSimple 5) and by the struct-returning shape
//   of 25 GetResourceAllocationInfo / 26 GetCustomHeapProperties. From slot 44 on the public order
//   does NOT hold: slot 48 returns an ID3D12CommandQueue from a 16-byte queue descriptor (4 calls),
//   slot 73 an ID3D12GraphicsCommandList from (16-byte descriptor, allocator, PSO, riid, out)
//   (3 calls), slots 55/57/67/86 return ID3D12PipelineState objects, 101 an ID3D12StateObject,
//   127 an ID3D12Resource with the CreateCommittedResource3 argument shape (heap properties, heap
//   flags, DESC1, initial layout, clear value, protected session, castable formats, riid, out) and
//   133 an ID3D12Resource from (address, desc, initial state, clear value, riid, out) (5 calls).
//   Slots 49, 50, 61, 62, 70, 71, 74, 78, 79, 80, 81, 108, 119, 132 are Xbox-specific with no public
//   counterpart; G27 established them (d3d12_xs_xbox.h). Unestablished slots stay logging stubs.
// * Struct returns: at slots 25/26 (and Xbox slot 108) the caller passes the hidden result pointer
//   in rcx and `this` in rdx (C-style), not `this` first as MSVC member functions do. Struct-
//   returning methods here accept both orders (SretMethods below) and count which one they saw.
//   An 8-byte struct (a descriptor handle) called with `this` in rcx is returned in rax: the
//   executable calls descriptor-heap slot 9 that way (rcx = this only, rax used).
// * Inlined accessors: 11 functions compare the byte at [device + 0x18] against 0 before writing
//   descriptors straight into descriptor memory; a non-zero byte makes them call
//   CopyDescriptorsSimple / CreateConstantBufferView instead. The device payload sets it to 1.
// * ID3D12CommandQueue: GetTimestampFrequency is slot 18 (out u64, result divided into a float),
//   Signal is slot 16 (fence, value), i.e. the public slots shifted by +2: two extra methods sit
//   between slots 3 and 15, so ExecuteCommandLists (public 10) is NOT established and stays a stub.
//   The submission thread executes command lists through slot 35 (this, count, lists, frame token,
//   flags) and presents through slot 32 (plane count, plane array, parameters); no other execute
//   slot is called. Slots 23/24/31: d3d12_xs_xbox.h.
// * ID3D12GraphicsCommandList: created by device slot 73 and immediately closed through slot 9
//   (public Close). G27 followed the lists through the context objects that own them: the Xbox
//   order (queries at 28..31, public 28..51 shifted by +4) and the inlined fast paths that write
//   command buffers directly are documented in d3d12_xs_xbox.h.
// * Allocators, descriptor heaps, fences, root signatures: only Release (slot 2) was observed on
//   tracked pointers; their method slots follow the public order (low confidence).
//
// Lifetime: bridge refcounting; children hold a com::Ref to their device. Unimplemented slots of
// every interface are logging stubs with SlotPolicy::ReturnNotImpl (first call logged with the slot
// index, every call counted in com::stub_hit_report()).

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <tuple>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/d3d12_guest_structs.h"
#include "gpu/d3d12_readback.h"
#include "gpu/d3d12_xs_dump.h"
#include "gpu/d3d12_xs_xbox.h"
#include "gpu/descriptor_heap.h"
#include "gpu/flush_page_index.h"
#include "gpu/log_rate_limiter.h"
#include "gpu/pso_cache.h"
#include "gpu/shader_pack_index.h"
#include "gpu/storage_reuse_lru.h"
#include "gpu/small_texture_pool.h"
#include "gpu/stage_reflection.h"
#include "gpu/state_bridge.h"
#include "guest/com_bridge.h"
#include "os/guest_window.h"
#include "os/vmem.h"

namespace playgta6::gpu::d3d12_guest {

namespace com = playgta6::guest::com;
namespace d3d = playgta6::gpu::d3d12_state;
using com::GuestPtr;
using com::Guid;
using com::HResult;
using com::InterfaceArg;
using com::OutPtr;

// ---------------------------------------------------------------------------------------------
// HRESULTs and IIDs (public values)
// ---------------------------------------------------------------------------------------------

inline constexpr HResult kSOk = 0;
inline constexpr HResult kSFalse = 1;
inline constexpr HResult kEInvalidArg = static_cast<HResult>(0x80070057u);
inline constexpr HResult kEOutOfMemory = static_cast<HResult>(0x8007000Eu);
inline constexpr HResult kEPointer = static_cast<HResult>(0x80004003u);
inline constexpr HResult kENoInterface = static_cast<HResult>(0x80004002u);
inline constexpr HResult kENotImpl = static_cast<HResult>(0x80004001u);
inline constexpr HResult kEFail = static_cast<HResult>(0x80004005u);
inline constexpr HResult kDxgiErrorNotFound = static_cast<HResult>(0x887A0002u);
inline constexpr HResult kDxgiErrorMoreData = static_cast<HResult>(0x887A0003u);

inline constexpr Guid IID_ID3D12Object = com::make_guid("c4fec28f-7966-4e95-9f94-f431cb56c3b8");
inline constexpr Guid IID_ID3D12DeviceChild = com::make_guid("905db94b-a00c-4140-9df5-2b64ca9ea357");
inline constexpr Guid IID_ID3D12Pageable = com::make_guid("63ee58fb-1268-4835-86da-f008ce62f0d6");
inline constexpr Guid IID_ID3D12Device = com::make_guid("189819f1-1db6-4b57-be54-1821339b85f7");
inline constexpr Guid IID_ID3D12Device11 = com::make_guid("5405c344-d457-444e-b4dd-2366e45aee39");
inline constexpr Guid IID_ID3D12CommandQueue = com::make_guid("0ec870a6-5d7e-4c22-8cfc-5baae07616ed");
inline constexpr Guid IID_ID3D12CommandAllocator = com::make_guid("6102dee4-af59-4b09-b999-b44d73f09b24");
inline constexpr Guid IID_ID3D12CommandList = com::make_guid("7116d91c-e7e4-47ce-b8c6-ec8168f437e5");
inline constexpr Guid IID_ID3D12GraphicsCommandList = com::make_guid("5b160d0f-ac1b-4185-8ba8-b3ae42a5a455");
inline constexpr Guid IID_ID3D12DescriptorHeap = com::make_guid("8efb471d-616c-4f49-90f7-127bb763fa51");
inline constexpr Guid IID_ID3D12RootSignature = com::make_guid("c54a6b66-72df-4ee8-8be5-a946a1429214");
inline constexpr Guid IID_ID3D12Fence = com::make_guid("0a753dcf-c4d8-4b91-adf6-be5a60d95a76");
inline constexpr Guid IID_ID3D12Fence1 = com::make_guid("433685fe-e22b-4ca0-a8db-b5b4f4dd0e4a");
inline constexpr Guid IID_ID3D12QueryHeap = com::make_guid("0d9658ae-ed45-469e-a61d-970ec583cab4");
inline constexpr Guid IID_ID3D12CommandSignature = com::make_guid("c36a797c-ec80-4f0a-8985-a7b2475082d1");
inline constexpr Guid IID_ID3D12Resource = com::make_guid("696442be-a72e-4059-bc79-5b5c98040fad");
inline constexpr Guid IID_ID3D12Heap = com::make_guid("6b3b2502-6e51-45b3-90ee-9884265e8df3");
inline constexpr Guid IID_ID3D12PipelineState = com::make_guid("765a30f3-f624-4c6f-a828-ace948622445");
inline constexpr Guid IID_ID3DBlob = com::make_guid("8ba5fb08-5195-40e2-ac58-0d989c3a0102");

// ---------------------------------------------------------------------------------------------
// Struct-returning methods (hidden result pointer in either rcx or rdx, see the header comment)
// ---------------------------------------------------------------------------------------------

struct SretStats {
    uint64_t this_first = 0;    // rcx = this, rdx = result (MSVC member convention)
    uint64_t result_first = 0;  // rcx = result, rdx = this (observed in the executable)
};
SretStats sret_stats();
void reset_sret_stats();

namespace detail {
void note_sret(bool this_first);
bool resolve_sret_this(GuestState& state, const void* host_tag, com::detail::Resolved* out, bool* this_first);

template <typename Host, auto M, typename R, typename... A, size_t... I>
void invoke_sret(com::CallContext& ctx, Host* self, uint64_t hidden, std::tuple<A...>*, std::index_sequence<I...>) {
    R result{};
    com::detail::marshal_call<void, A...>(
        ctx, 2u, [&](auto&&... args) { result = (self->*M)(std::forward<decltype(args)>(args)...); },
        std::index_sequence<I...>{});
    if (!com::write_guest(hidden, result)) {
        com::detail::log_bad_pointer("hidden result pointer", hidden);
    }
    ctx.set_rax(hidden);
}

template <typename Host, auto M>
void sret_thunk(void* raw_state) {
    using Traits = com::detail::MemberFn<decltype(M)>;
    using R = typename Traits::Ret;
    using Params = typename Traits::Params;
    static_assert(std::is_class_v<R> && std::is_trivially_copyable_v<R>, "struct-returning methods only");
    GuestState& state = *static_cast<GuestState*>(raw_state);
    com::detail::Resolved r;
    bool this_first = true;
    resolve_sret_this(state, com::detail::type_tag<Host>(), &r, &this_first);
    com::GuestStateScope scope(state);
    const uint64_t self = this_first ? state.gpr[GuestReg::Rcx] : state.gpr[GuestReg::Rdx];
    const uint64_t hidden = this_first ? state.gpr[GuestReg::Rdx] : state.gpr[GuestReg::Rcx];
    com::CallContext ctx{state, self, r.object.get(), r.iface, state.gpr[GuestReg::Rsp]};
    if constexpr (sizeof(R) <= 8 && std::tuple_size_v<Params> == 0) {
        // An 8-byte struct (a descriptor handle) with `this` in rcx comes back in rax: the
        // executable calls GetCPUDescriptorHandleForHeapStart with rcx = this only and uses rax
        // (rdx holds whatever the caller left there). With the result pointer in rcx and `this`
        // in rdx it is a hidden result, as below.
        if (this_first) {
            R result = (static_cast<Host*>(r.host)->*M)();
            uint64_t bits = 0;
            std::memcpy(&bits, &result, sizeof(R));
            ctx.set_rax(bits);
            com::detail::pop_return(state);
            return;
        }
    }
    invoke_sret<Host, M, R>(ctx, static_cast<Host*>(r.host), hidden, static_cast<Params*>(nullptr),
                            std::make_index_sequence<std::tuple_size_v<Params>>{});
    com::detail::pop_return(state);
}
}  // namespace detail

template <typename Host>
struct Methods : com::Methods<Host> {
    template <auto M>
    static constexpr com::MethodDesc sret(const char* name) {
        return com::MethodDesc{name, &detail::sret_thunk<Host, M>};
    }
};

// ---------------------------------------------------------------------------------------------
// Host interfaces
// ---------------------------------------------------------------------------------------------

// ID3D12Object (slots 3..6).
struct IObjectHost {
    virtual HResult GetPrivateData(const Guid& guid, GuestPtr<uint32_t> data_size, uint64_t data) = 0;
    virtual HResult SetPrivateData(const Guid& guid, uint32_t data_size, uint64_t data) = 0;
    virtual HResult SetPrivateDataInterface(const Guid& guid, uint64_t unknown) = 0;
    virtual HResult SetName(uint64_t name) = 0;
};

// ID3D12DeviceChild (slot 7).
struct IDeviceChildHost : IObjectHost {
    virtual HResult GetDevice(const Guid& riid, OutPtr device) = 0;
};

class Device;
class Fence;

// Private data / name storage shared by every object (ID3D12Object semantics).
class PrivateData {
public:
    HResult get(const Guid& guid, GuestPtr<uint32_t> data_size, uint64_t data);
    HResult set(const Guid& guid, uint32_t data_size, uint64_t data);
    HResult set_interface(const Guid& guid, uint64_t unknown);
    HResult set_name(uint64_t name);
    std::u16string name() const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, std::vector<uint8_t>> entries_;  // key: raw GUID bytes
    std::u16string name_;
};

template <typename Iface>
class ObjectBase : public com::ComObject, public Iface {
public:
    HResult GetPrivateData(const Guid& guid, GuestPtr<uint32_t> size, uint64_t data) override {
        return data_.get(guid, size, data);
    }
    HResult SetPrivateData(const Guid& guid, uint32_t size, uint64_t data) override {
        return data_.set(guid, size, data);
    }
    HResult SetPrivateDataInterface(const Guid& guid, uint64_t unknown) override {
        return data_.set_interface(guid, unknown);
    }
    HResult SetName(uint64_t name) override { return data_.set_name(name); }
    std::u16string debug_name() const { return data_.name(); }

private:
    PrivateData data_;
};

template <typename Iface>
class ChildBase : public ObjectBase<Iface> {
public:
    explicit ChildBase(com::Ref<Device> device) : device_(std::move(device)) {}
    HResult GetDevice(const Guid& riid, OutPtr out) override;
    Device& device() const { return *device_; }

protected:
    com::Ref<Device> device_;
};

// ID3DBlob: 3 GetBufferPointer, 4 GetBufferSize.
struct IBlobHost {
    virtual uint64_t GetBufferPointer() = 0;
    virtual uint64_t GetBufferSize() = 0;
};

// ID3D12CommandAllocator: 8 Reset; Xbox 9 (callback, context) (see d3d12_xs_xbox.h).
struct ICommandAllocatorHost : IDeviceChildHost {
    virtual HResult Reset() = 0;
    virtual HResult SetAllocationCallback_unverified(uint64_t callback, uint64_t context) = 0;
};

// ID3D12Fence: 8 GetCompletedValue, 9 SetEventOnCompletion, 10 Signal; ID3D12Fence1: 11 GetCreationFlags.
struct IFenceHost : IDeviceChildHost {
    virtual uint64_t GetCompletedValue() = 0;
    virtual HResult SetEventOnCompletion(uint64_t value, uint64_t event_handle) = 0;
    virtual HResult Signal(uint64_t value) = 0;
    virtual uint32_t GetCreationFlags() = 0;
    // Xbox slot 12: () -> the guest address of a 64-bit word holding the completed value. The
    // executable's fence wait (0x144FB0AD0) reads the value through it, compares it with
    // UINT64_MAX (device removed) and with the target value, and waits while it is below.
    virtual uint64_t CompletedValueAddress() = 0;
};

// ID3D12DescriptorHeap: 8 GetDesc, 9 GetCPUDescriptorHandleForHeapStart, 10 GetGPUDescriptorHandleForHeapStart.
struct CpuHandleResult {
    uint64_t ptr;
};
struct GpuHandleResult {
    uint64_t ptr;
};
struct IDescriptorHeapHost : IDeviceChildHost {
    virtual GuestDescriptorHeapDesc GetDesc() = 0;
    virtual CpuHandleResult GetCPUDescriptorHandleForHeapStart() = 0;
    virtual GpuHandleResult GetGPUDescriptorHandleForHeapStart() = 0;
};

// ID3D12RootSignature, ID3D12QueryHeap, ID3D12CommandSignature: no methods beyond ID3D12DeviceChild.
struct IRootSignatureHost : IDeviceChildHost {};
struct IQueryHeapHost : IDeviceChildHost {};
struct ICommandSignatureHost : IDeviceChildHost {};

// ID3D12CommandQueue in the Xbox slot order (see the header comment).
struct ICommandQueueHost : IDeviceChildHost {
    virtual HResult Signal(InterfaceArg<IFenceHost> fence, uint64_t value) = 0;                // 16
    virtual HResult Wait(InterfaceArg<IFenceHost> fence, uint64_t value) = 0;                  // 17
    virtual HResult GetTimestampFrequency(GuestPtr<uint64_t> frequency) = 0;                   // 18
    virtual HResult PresentX(uint32_t plane_count, uint64_t planes, uint64_t parameters) = 0;  // 32
    virtual void ExecuteCommandListsX(com::CallContext& ctx) = 0;                              // 35
    // d3d12_xs_xbox.cpp
    virtual HResult SuspendX(uint32_t flags) = 0;                                              // 23
    virtual HResult ResumeX() = 0;                                                             // 24
    virtual HResult Slot31_UpdatePageMappings_unverified(com::CallContext& ctx) = 0;           // 31
};

// D3D12_RESOURCE_DESC1 as a struct-return value (72 bytes).
struct ResourceDesc1Result {
    uint8_t bytes[kGuestResourceDesc1Size];
};

// ID3D12Resource: 8 Map, 9 Unmap, 10 GetDesc, 11 GetGPUVirtualAddress, 12 WriteToSubresource,
// 13 ReadFromSubresource, 14 GetHeapProperties; ID3D12Resource2: 16 GetDesc1.
struct IResourceHost : IDeviceChildHost {
    virtual HResult Map(uint32_t subresource, uint64_t read_range, OutPtr data) = 0;
    virtual void Unmap(uint32_t subresource, uint64_t written_range) = 0;
    virtual GuestResourceDesc GetDesc() = 0;
    virtual uint64_t GetGPUVirtualAddress() = 0;
    virtual HResult WriteToSubresource(com::CallContext& ctx) = 0;
    virtual HResult ReadFromSubresource(com::CallContext& ctx) = 0;
    virtual HResult GetHeapProperties(GuestPtr<GuestHeapProperties> properties, GuestPtr<uint32_t> flags) = 0;
    virtual ResourceDesc1Result GetDesc1() = 0;
};

// ID3D12Heap: 8 GetDesc.
struct IHeapHost : IDeviceChildHost {
    virtual GuestHeapDesc GetDesc() = 0;
};

// ID3D12PipelineState: 8 GetCachedBlob.
struct IPipelineStateHost : IDeviceChildHost {
    virtual HResult GetCachedBlob(OutPtr blob) = 0;
};

// ID3D12GraphicsCommandList. The slot comments give the public order; the Xbox vtable
// (d3d12_guest_command_list.cpp, evidence in d3d12_xs_xbox.h) keeps 0..27, puts the query and
// predication methods at 28..31, shifts public 28..51 by +4 and has Xbox-only slots further up.
struct IGraphicsCommandListHost : IDeviceChildHost {
    virtual uint32_t GetType() = 0;                                                                  // 8
    virtual HResult Close() = 0;                                                                     // 9
    virtual HResult Reset(InterfaceArg<ICommandAllocatorHost> allocator,
                          InterfaceArg<IPipelineStateHost> initial_state) = 0;                       // 10
    // Xbox slot 139: (allocator, initial state, u64) -> HRESULT, a Reset with a third argument.
    virtual HResult ResetX(InterfaceArg<ICommandAllocatorHost> allocator, InterfaceArg<IPipelineStateHost> initial_state,
                           uint64_t extra) = 0;
    virtual void ClearState(InterfaceArg<IPipelineStateHost> state) = 0;                             // 11
    virtual void DrawInstanced(uint32_t vertices, uint32_t instances, uint32_t start_vertex,
                               uint32_t start_instance) = 0;                                         // 12
    virtual void DrawIndexedInstanced(uint32_t indices, uint32_t instances, uint32_t start_index,
                                      int32_t base_vertex, uint32_t start_instance) = 0;             // 13
    virtual void Dispatch(uint32_t x, uint32_t y, uint32_t z) = 0;                                   // 14
    virtual void CopyBufferRegion(InterfaceArg<IResourceHost> dst, uint64_t dst_offset,
                                  InterfaceArg<IResourceHost> src, uint64_t src_offset,
                                  uint64_t bytes) = 0;                                               // 15
    virtual void CopyTextureRegion(com::CallContext& ctx) = 0;                                       // 16
    // Xbox slot 85: (pDst, DstX, DstY, DstZ, pSrc, pSrcBox, flags), CopyTextureRegion + trailing flags.
    virtual void CopyTextureRegionX(com::CallContext& ctx) = 0;
    virtual void CopyResource(InterfaceArg<IResourceHost> dst, InterfaceArg<IResourceHost> src) = 0;  // 17
    virtual void ResolveSubresource(com::CallContext& ctx) = 0;                                      // 19
    virtual void IASetPrimitiveTopology(uint32_t topology) = 0;                                      // 20
    virtual void RSSetViewports(uint32_t count, uint64_t viewports) = 0;                             // 21
    virtual void RSSetScissorRects(uint32_t count, uint64_t rects) = 0;                              // 22
    virtual void OMSetBlendFactor(uint64_t factor) = 0;                                              // 23
    virtual void OMSetStencilRef(uint32_t ref) = 0;                                                  // 24
    virtual void SetPipelineState(InterfaceArg<IPipelineStateHost> state) = 0;                       // 25
    virtual void ResourceBarrier(uint32_t count, uint64_t barriers) = 0;                             // 26
    virtual void SetDescriptorHeaps(uint32_t count, uint64_t heaps) = 0;                             // 28
    virtual void SetComputeRootSignature(InterfaceArg<IRootSignatureHost> signature) = 0;            // 29
    virtual void SetGraphicsRootSignature(InterfaceArg<IRootSignatureHost> signature) = 0;           // 30
    virtual void SetComputeRootDescriptorTable(uint32_t index, uint64_t base) = 0;                   // 31
    virtual void SetGraphicsRootDescriptorTable(uint32_t index, uint64_t base) = 0;                  // 32
    virtual void SetComputeRoot32BitConstant(uint32_t index, uint32_t value, uint32_t offset) = 0;   // 33
    virtual void SetGraphicsRoot32BitConstant(uint32_t index, uint32_t value, uint32_t offset) = 0;  // 34
    virtual void SetComputeRoot32BitConstants(uint32_t index, uint32_t count, uint64_t values,
                                              uint32_t offset) = 0;                                  // 35
    virtual void SetGraphicsRoot32BitConstants(uint32_t index, uint32_t count, uint64_t values,
                                               uint32_t offset) = 0;                                 // 36
    virtual void SetComputeRootConstantBufferView(uint32_t index, uint64_t va) = 0;                  // 37
    virtual void SetGraphicsRootConstantBufferView(uint32_t index, uint64_t va) = 0;                 // 38
    virtual void SetComputeRootShaderResourceView(uint32_t index, uint64_t va) = 0;                  // 39
    virtual void SetGraphicsRootShaderResourceView(uint32_t index, uint64_t va) = 0;                 // 40
    virtual void SetComputeRootUnorderedAccessView(uint32_t index, uint64_t va) = 0;                 // 41
    virtual void SetGraphicsRootUnorderedAccessView(uint32_t index, uint64_t va) = 0;                // 42
    virtual void IASetIndexBuffer(uint64_t view) = 0;                                                // 43
    virtual void IASetVertexBuffers(uint32_t start, uint32_t count, uint64_t views) = 0;             // 44
    virtual void SOSetTargets(uint32_t start, uint32_t count, uint64_t views) = 0;                   // 45
    virtual void OMSetRenderTargets(uint32_t count, uint64_t rtvs, bool single_range, uint64_t dsv) = 0;  // 46
    virtual void ClearDepthStencilView(uint64_t dsv, uint32_t flags, float depth, uint8_t stencil,
                                       uint32_t rect_count, uint64_t rects) = 0;                     // 47
    virtual void ClearRenderTargetView(uint64_t rtv, uint64_t color, uint32_t rect_count,
                                       uint64_t rects) = 0;                                          // 48
    virtual void ClearUnorderedAccessView(com::CallContext& ctx) = 0;                                // 49, 50
    virtual void DiscardResource(InterfaceArg<IResourceHost> resource, uint64_t region) = 0;         // 51
    virtual void BeginQuery(com::CallContext& ctx) = 0;                                              // 52
    virtual void EndQuery(com::CallContext& ctx) = 0;                                                // 53
    virtual void ResolveQueryData(com::CallContext& ctx) = 0;                                        // 54
    virtual void SetPredication(com::CallContext& ctx) = 0;                                          // 55
    virtual void DebugMarker(com::CallContext& ctx) = 0;                                             // 56..58
    // Xbox slots (see d3d12_xs_xbox.h).
    virtual void SetPipelineStateX(com::CallContext& ctx) = 0;  // Xbox 25: staged PSO, rdx fallback
    virtual void ExecuteIndirect(com::CallContext& ctx) = 0;    // Xbox 61
    virtual void OMSetDepthBounds(float min_depth, float max_depth) = 0;               // Xbox 70
    virtual void RSSetShadingRate(uint32_t rate, uint64_t combiners) = 0;              // Xbox 126
    virtual void RSSetShadingRateImage(InterfaceArg<IResourceHost> image) = 0;         // Xbox 127
    // Xbox 71: (address, u32 value, 0, 0): writes a 32-bit value to guest memory when the GPU
    // reaches this point (the title's per-frame marker array, 3 frames x 18 u32).
    virtual void WriteValue32(uint64_t address, uint32_t value, uint32_t a3, uint32_t a4) = 0;
    // Xbox 74: (address, 7, value, 1): writes a 64-bit value (a frame counter, a word the CPU
    // also writes with a 64-bit store) to guest memory when the GPU reaches this point.
    virtual void WriteValue64(uint64_t address, uint32_t a2, uint64_t value, uint32_t a4) = 0;
    // Xbox 146: enhanced barriers, (UINT32 NumBarrierGroups, const D3D12_BARRIER_GROUP*).
    virtual void Barrier(uint32_t group_count, uint64_t groups) = 0;
    // Xbox-only slots whose meaning is not established: recorded, no effect.
    virtual void note_xbox_slot(uint32_t slot) = 0;
    template <uint32_t Slot>
    void XboxSlot(com::CallContext&) {
        note_xbox_slot(Slot);
    }
};

struct HeapPropertiesResult {
    GuestHeapProperties props;
};

// ID3D12Device (public slots 0..43 plus the evidenced Xbox slots).
struct IDeviceHost : IObjectHost {
    virtual uint32_t GetNodeCount() = 0;                                                                  // 7
    virtual HResult CreateCommandQueue(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) = 0;  // 8
    virtual HResult CreateCommandAllocator(uint32_t type, const Guid& riid, OutPtr out) = 0;               // 9
    virtual HResult CheckFeatureSupport(uint32_t feature, uint64_t data, uint32_t size) = 0;               // 13
    virtual HResult CreateDescriptorHeap(const GuestDescriptorHeapDesc* desc, const Guid& riid, OutPtr out) = 0;  // 14
    virtual uint32_t GetDescriptorHandleIncrementSize(uint32_t type) = 0;                                 // 15
    virtual HResult CreateRootSignature(uint32_t node_mask, uint64_t blob, uint64_t length, const Guid& riid,
                                        OutPtr out) = 0;                                                    // 16
    virtual void CreateConstantBufferView(uint64_t desc, uint64_t dest) = 0;                              // 17
    virtual void CreateSampler(uint64_t desc, uint64_t dest) = 0;                                         // 22
    virtual void CopyDescriptors(com::CallContext& ctx) = 0;                                              // 23
    virtual void CopyDescriptorsSimple(uint32_t count, uint64_t dest, uint64_t src, uint32_t type) = 0;   // 24
    virtual GuestResourceAllocationInfo GetResourceAllocationInfo(uint32_t visible_mask, uint32_t count,
                                                                  uint64_t descs) = 0;                     // 25
    virtual HeapPropertiesResult GetCustomHeapProperties(uint32_t node_mask, uint32_t heap_type) = 0;     // 26
    virtual HResult MakeResident(uint32_t count, uint64_t objects) = 0;                                   // 34
    virtual HResult Evict(uint32_t count, uint64_t objects) = 0;                                          // 35
    virtual HResult CreateFence(uint64_t initial_value, uint32_t flags, const Guid& riid, OutPtr out) = 0;  // 36
    virtual HResult GetDeviceRemovedReason() = 0;                                                         // 37
    virtual void GetCopyableFootprints(com::CallContext& ctx) = 0;                                        // 38
    virtual HResult CreateQueryHeap(const GuestQueryHeapDesc* desc, const Guid& riid, OutPtr out) = 0;    // 39
    virtual HResult SetStablePowerState(bool enable) = 0;                                                 // 40
    virtual HResult CreateCommandSignature(const GuestCommandSignatureDesc* desc, uint64_t root_signature,
                                           const Guid& riid, OutPtr out) = 0;                               // 41
    virtual HResult CreateCommandQueueX(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) = 0;  // 48
    // Resources, PSOs, command lists, views (d3d12_guest_resources.cpp, d3d12_guest_command_list.cpp).
    virtual HResult CreateGraphicsPipelineState(uint64_t desc, const Guid& riid, OutPtr out) = 0;          // 10
    virtual HResult CreateComputePipelineState(uint64_t desc, const Guid& riid, OutPtr out) = 0;           // 11
    virtual HResult CreateCommandList(uint32_t node_mask, uint32_t type, InterfaceArg<ICommandAllocatorHost> allocator,
                                      InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid,
                                      OutPtr out) = 0;                                                     // 12
    virtual void CreateShaderResourceView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) = 0;  // 18
    virtual void CreateUnorderedAccessView(InterfaceArg<IResourceHost> resource, InterfaceArg<IResourceHost> counter,
                                           uint64_t desc, uint64_t dest) = 0;                             // 19
    virtual void CreateRenderTargetView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) = 0;  // 20
    virtual void CreateDepthStencilView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) = 0;  // 21
    virtual HResult CreateCommittedResource(com::CallContext& ctx) = 0;                                   // 27
    virtual HResult CreateHeap(const GuestHeapDesc* desc, const Guid& riid, OutPtr out) = 0;              // 28
    virtual HResult CreatePlacedResource(com::CallContext& ctx) = 0;                                      // 29
    virtual HResult CreateReservedResource(uint64_t desc, uint32_t state, uint64_t clear, const Guid& riid,
                                           OutPtr out) = 0;                                                 // 30
    virtual HResult CreateCommandListX(const GuestCommandListDescX* desc, InterfaceArg<ICommandAllocatorHost> allocator,
                                       InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid,
                                       OutPtr out) = 0;                                                    // 73
    virtual HResult CreateCommittedResource3(com::CallContext& ctx) = 0;                                  // 127
    virtual HResult CreatePlacedResourceX(com::CallContext& ctx) = 0;                                     // 133
    // Xbox slots (d3d12_xs_xbox.cpp; evidence and confidence in d3d12_xs_xbox.h).
    virtual HResult Slot49_SetFeatureToggle_unverified(uint32_t feature, uint32_t value) = 0;           // 49
    virtual HResult Slot50_RegisterRemovalCallbacks_unverified(uint64_t a, uint64_t b, uint64_t c) = 0;  // 50
    virtual HResult CreatePipelineStateWithOverrides(InterfaceArg<IPipelineStateHost> base, uint32_t count,
                                                     uint64_t overrides, const Guid& riid, OutPtr out) = 0;  // 55
    virtual HResult CreateNativePipelineState(uint64_t desc, const Guid& riid, OutPtr out) = 0;          // 57
    virtual HResult Slot61_unverified(uint64_t a, uint64_t b) = 0;                                       // 61
    virtual HResult CreateComputePipelineStateX(com::CallContext& ctx) = 0;                              // 62
    virtual HResult CreatePipelineStateFromBlob_unverified(com::CallContext& ctx) = 0;                   // 67
    virtual void CreateShaderResourceViewAtAddress(uint64_t resource_desc, uint64_t view_desc, uint64_t dest) = 0;  // 70
    virtual void CreateUnorderedAccessViewAtAddress(uint64_t resource_desc, uint64_t view_desc,
                                                    uint64_t dest) = 0;                                  // 71
    virtual HResult RegisterPagePool_unverified(uint64_t address, uint32_t pages, GuestPtr<uint64_t> handle) = 0;  // 74
    virtual HResult SetFrameIntervalX(uint64_t output, uint32_t interval_us, uint32_t period, uint32_t flags) = 0;  // 78
    virtual HResult ScheduleFrameEventX(uint32_t type, uint32_t offset_us, uint64_t reserved, uint32_t flags) = 0;  // 79
    virtual HResult WaitFrameEventX(uint32_t type, uint32_t timeout_ms, uint64_t reserved, uint32_t flags,
                                    GuestPtr<uint64_t> token) = 0;                                       // 80
    virtual HResult QueryFrameStatistics_unverified(uint64_t token, uint32_t flags, GuestPtr<uint32_t> count,
                                                    uint64_t entries) = 0;                               // 81
    virtual HResult CreatePipelineStateFromStream(uint64_t stream_desc, const Guid& riid, OutPtr out) = 0;  // 86
    virtual HResult CreateStateObject(uint64_t desc, const Guid& riid, OutPtr out) = 0;                  // 101
    virtual GuestResourceAllocationInfo GetResourceAllocationInfo1(uint32_t visible_mask, uint32_t count, uint64_t descs,
                                                                   uint64_t infos) = 0;                  // 108
    virtual HResult Slot119_unverified(uint64_t buffer, uint32_t size, uint32_t flags) = 0;              // 119
    virtual HResult Slot132_unverified(uint32_t value) = 0;                                              // 132
};

// ---------------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------------

// Where packets go (default: playgta6::os::wgpu_submit). Called with the device's submit mutex held.
using PacketSink = std::function<void(const uint8_t* data, size_t size)>;

// Pluggable detiling callback for console-tiled guest textures (G49b / G49a hook).
// Given raw guest bytes (`src`, `src_bytes`) in console layout `req.layout` (e.g. 0x100 | swizzle),
// writes the linear rows of subresource (`req.mip`, `req.array_slice`) with `req.dst_row_pitch`
// bytes per row into `dst` (`dst_bytes`). `src` is the whole surface (every mip and slice) from
// the resource's base address. Returns true on success, false if unsupported.
struct TextureDetileRequest {
    uint32_t dxgi_format = 0;
    uint32_t width = 0;   // mip 0
    uint32_t height = 0;
    uint32_t depth = 1;   // 3D depth (1 for 1D/2D)
    uint32_t mip_levels = 1;
    uint32_t layout = 0;
    uint32_t block_bytes = 0;
    uint32_t block_width = 1;
    uint32_t block_height = 1;
    uint32_t dst_row_pitch = 0;
    uint32_t array_size = 1;   // 1D/2D array slices
    uint32_t mip = 0;          // subresource to write
    uint32_t array_slice = 0;
};
using DetileFn = std::function<bool(const TextureDetileRequest& req,
                                    const uint8_t* src,
                                    size_t src_bytes,
                                    uint8_t* dst,
                                    size_t dst_bytes)>;

// Dirty-tracking state and layout metadata for a guest-memory-backed GPU texture (G49b).
struct TrackedTexture {
    uint32_t resource_id = 0;
    uint64_t gpu_va = 0;
    uint32_t dxgi_format = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint32_t layout = 0;
    uint32_t dimension = 0;     // ResourceDimension (2=1D, 3=2D, 4=3D)
    uint32_t sample_count = 1;
    uint64_t last_upload_hash = 0;  // whole source surface
    bool ever_uploaded = false;
    uint32_t cap_floor_applied = 0;  // top mip levels dropped by the texture size cap (gpu/texture_budget.h)
    // Host reallocation under the cap (gpu/texture_budget.h): `create` is the packet the full-size host
    // texture was made with (has_create), `host_floor` the top levels the host texture currently
    // lacks (it is allocated at the capped size), `host_saved_bytes` the bytes that saves.
    bool has_create = false;
    GpuCreateTexture2Payload create{};
    uint32_t host_floor = 0;
    uint64_t host_saved_bytes = 0;
    bool streamable = false;  // created as a reserved (tiled) resource: unmapped levels are not loaded
    // Submission (Device::begin_submission) that last looked at this texture: a texture bound in
    // several bind groups or lists of one ExecuteCommandLists is read and hashed once.
    uint64_t synced_submission = 0;
    // Per subresource (mip + array_slice * mip_levels): hash of the last uploaded linear texels,
    // 0 = never uploaded (the GPU texture is still zero there).
    std::vector<uint64_t> subresource_hashes;
    // Per subresource: hashes of its bands of kUploadBandRows block rows at the last upload (2D
    // subresources taller than one band; empty otherwise). A changed subresource writes only the
    // bands whose hash changed (#691).
    std::vector<std::vector<uint64_t>> subresource_bands;
    // Window write-mark tracking (G68, #663): when the texture lives in the translated CPU window,
    // [window_indexed_va, window_indexed_va + window_indexed_size) is indexed in
    // Device::texture_slot_index_. flush_cpu_buffers sets window_dirty whenever a dirty slot
    // overlaps it (sharing the submission's MarkPass with CPU buffers); sync_bound_texture_memory
    // skips reading and hashing when window_checked is true, window_dirty is false, and no slot is
    // currently marked.
    uint64_t window_indexed_va = 0;
    uint64_t window_indexed_size = 0;
    bool window_indexed = false;
    bool window_checked = false;
    bool window_dirty = true;
    // G75 (#663): consecutive full-surface (>= 75% bands changed) dirty frames in a translated
    // window. Once >= 3, sync_bound_texture_memory skips per-band content_hash() while dirty marks
    // keep arriving, and resets to 0 on any clean/unmarked or partial-update frame.
    uint8_t streaming_full_frames = 0;
    // Native shared-view write tracking (no translated window): the range the cursor below watches
    // and whether the last complete check (zero scan or hash) of it is still current. Cleared when
    // os::take_written_view_pages reports pages written, or does not cover the range.
    const uint8_t* view_host = nullptr;
    uint64_t view_size = 0;
    os::WriteCursor view_cursor{};
    bool view_checked = false;
    uint32_t view_checked_cap = 0;  // texture budget cap in force when view_checked was set
};
// Block rows per dirty-tracked band of a 2D texture upload (TrackedTexture::subresource_bands).
constexpr uint32_t kUploadBandRows = 64;

struct DeviceOptions {
    PacketSink sink;                    // empty: os::wgpu_submit
    bridge::BridgeOptions bridge;       // id_allocator is replaced by the device's allocator
    uint64_t timestamp_frequency = 1000000000ull;
    // Xbox slot 57: index the console-native pipeline's stages, then fail the call (E_NOTIMPL) so
    // the title creates the pipeline from DXIL through slot 86 when it is requested (d3d12_xs_xbox.h,
    // "Native pipelines and their DXIL"). On by default. This deliberately differs from a console,
    // which creates these pipelines: the fallback is the title's own code path (the same one it
    // takes for compute when slot 67 is refused), and only it carries the public state, input
    // layouts and shader signatures the WebGPU side needs. Off: a native pipeline state is
    // created (unresolved; its draws are skipped). Boot harness: --native-pipelines turns it off.
    bool refuse_native_pipelines = true;
    // Pipeline dump directory (d3d12_xs_dump.h); empty: no dump.
    std::string dump_dir;
    // Keys of the WGSL pack the WebGPU side loads (shader_pack_index.h; boot harness:
    // --shader-pack). Graphics pipelines whose stages all have a key are resolved: see
    // Device::resolve_with_pack. Null: no pack, every pipeline stays unresolved.
    std::shared_ptr<const ShaderPackIndex> shader_pack;
    // Reflection of the pack's stages (gpu/stage_reflection.h; boot harness: <pack>.reflect.json).
    // Pack-resolved pipelines whose stages all have an entry get their vertex attributes and
    // pipeline layout from it (join_vertex_inputs / join_pipeline_layout); a pipeline that cannot be
    // joined stays unresolved (its draws are skipped), with one report per pipeline. Stages without
    // an entry keep layout_from_root_signature and element i -> @location(i).
    std::shared_ptr<const StageReflectionIndex> stage_reflection;
    // Optional detiling hook for tiled console layouts (layout >= 0x101).
    DetileFn detile_texture;
    // Fence completion latency (R40, #663 / #1500): by default (`fence_publish_lag = false`) a
    // queue Signal's value is published at issue, when its work has been handed off. The lag below
    // guarded #691's grey frames, whose real causes were #1503 / #1507; it held the render worker's
    // completed-value poll (guest 0x144fb0c27) for up to two presents (~15 presents/s). It also hid
    // the menu-text corruption of publishing at issue: storage relocation homes handed to another
    // range while still bound (relocate_storage_buffer, refresh_memory_bind_groups; #663).
    // With `fence_publish_lag = true` (boot harness: --fence-lag) the old rule applies: a queue
    // Signal's value is published (the fence's completed-value word and GetCompletedValue) once
    // `present_epoch` has advanced by `fence_publish_lag_frames` presents (default 1: on the frame's
    // own PresentX, after all of its ExecuteCommandLists have already snapshotted upload-ring
    // memory), or after
    // `fence_publish_cap_ms` without a present (default 33 ms, one 30 Hz frame). Holding signals
    // within the submitting frame's epoch prevents mid-frame overwrites between earlier and later
    // ExecuteCommandLists, while publishing on that frame's PresentX avoids the 2-present (~66 ms)
    // 15 FPS stall when the render worker polls the completed-value word. Set
    // `fence_publish_lag_frames = 2` for the conservative next-frame lag, or `0` to publish at
    // issue. Blocking waits (fence events, a null-event SetEventOnCompletion, queue Wait) for an
    // executed value complete at once and publish it.
    bool fence_publish_lag = false;
    uint32_t fence_publish_lag_frames = 1;
    uint32_t fence_publish_cap_ms = 33;
    // Lazy GPU objects (G66, #1077): a guest resource's CreateBuffer / CreateTexture2 packet is held
    // until the first submission that references its id (a bind group entry, render pass attachment,
    // vertex / index buffer, WriteBuffer / WriteTexture) and written right before that packet; a
    // resource released before any use gets neither a create nor a DestroyResource packet. Titles
    // create thousands of resources (4K render targets, placed resources in large heaps) that are
    // never bound; creating them all at once costs the browser's GPU process gigabytes. Ids are
    // allocated at creation either way, so recorded packets stay valid. Off: created at once.
    bool defer_gpu_objects = true;
    // GPU -> CPU read-back (G66, d3d12_readback.h): copies into CPU-visible memory become
    // ReadbackBuffer / ReadbackTexture packets either way. With `gpu_readback_hold` a queue Signal
    // issued after one is held until the consumer reports the bytes written (never released early;
    // `gpu_readback_timeout_ms` only logs a warning per held signal). Default: on where a consumer answers (the browser build).
#ifdef __EMSCRIPTEN__
    bool gpu_readback_hold = true;
#else
    bool gpu_readback_hold = false;
#endif
    uint32_t gpu_readback_timeout_ms = 250;
};

// One PresentX plane of the plain canvas composition (Device::present): its texture and the host
// texture's format and size as registered when the plane is drawn.
struct PresentPlaneSource {
    uint32_t texture_id = 0;
    pso::WgpuTextureFormat format = pso::WgpuTextureFormat::Undefined;
    uint32_t width = 0;
    uint32_t height = 0;
};

// The composition's objects, created on first use: shader modules, layouts, sampler, one pipeline
// per (canvas format, blended) and one bind group per (texture, view format). A cached bind group
// is only reused for the same texture size and format; otherwise it is re-created and the old id
// is listed in `retired` for the caller to destroy.
struct PresentBlitCache {
    uint32_t vs = 0, fs = 0, group_layout = 0, pipeline_layout = 0, sampler = 0;
    std::unordered_map<uint32_t, uint32_t> pipelines;  // canvas format code << 1 | blended -> pipeline
    struct Group {
        uint32_t id = 0;
        uint32_t texture_id = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        pso::WgpuTextureFormat format = pso::WgpuTextureFormat::Undefined;
    };
    std::unordered_map<uint64_t, Group> groups;  // (texture id << 8) | view format -> bind group
    std::vector<uint32_t> retired;               // replaced bind groups, to destroy
    std::unordered_set<std::string> logged_layouts;  // compositions logged so far (each once, capped)
};

// One drawn plane of a composition (for the log and tests).
struct PresentPlaneDraw {
    uint32_t texture_id = 0;
    uint32_t width = 0, height = 0;
    pso::WgpuTextureFormat format = pso::WgpuTextureFormat::Undefined;
    uint8_t view_format = 0;  // 0: the texture's own format
    uint32_t bind_group = 0;
    bool blended = false;  // premultiplied alpha over the planes below (all but the bottom plane)
};

// Emits the canvas pass compositing `planes` given top first (PresentX order: plane 0 on top, the
// last plane at the bottom); planes that are not drawable (id 0, not a filterable colour format)
// are skipped. The bottom drawable plane is drawn opaque and picks the canvas format, the others
// are blended over it with premultiplied alpha. Returns the number of planes drawn; `draws`, when
// given, receives them bottom first.
uint32_t emit_present_plane_blit(PresentBlitCache& cache, std::span<const PresentPlaneSource> planes,
                                 const std::function<uint32_t()>& allocate_id, CommandEncoder& enc,
                                 std::vector<PresentPlaneDraw>* draws = nullptr);

struct DeviceStats {
    uint64_t resources = 0;
    uint64_t heaps = 0;
    uint64_t pipeline_states = 0;
    uint64_t command_lists = 0;
    uint64_t upload_bytes = 0;        // WriteBuffer / WriteTexture payload bytes emitted from guest memory
    uint64_t textures_uploaded = 0;   // WriteTexture packets emitted from guest texture memory
    uint64_t skipped_draws = 0;       // draws skipped because the bound PSO has no shader module
    uint64_t unsupported_copies = 0;  // GPU-to-GPU copies, resolves, ... (no packet; diagnosed)
    uint64_t queues = 0;
    uint64_t command_allocators = 0;
    uint64_t descriptor_heaps = 0;
    uint64_t root_signatures = 0;
    uint64_t fences = 0;
    uint64_t presents = 0;
    uint64_t submissions = 0;
    uint64_t bad_arguments = 0;  // malformed guest structs / pointers (diagnosed, E_INVALIDARG)
    // O36: the live CPU-visible buffers' shadows: the bytes they cover (an eager shadow's host
    // memory) and the bytes allocated for them (lazy chunks written so far).
    uint64_t shadow_bytes = 0;
    uint64_t shadow_resident_bytes = 0;
    // R40: bytes of CPU-visible buffers compared with their shadow (all of a buffer without write
    // tracking, only the pages written since the last flush with it).
    uint64_t shadow_compared_bytes = 0;
    // Lazy GPU objects (DeviceOptions::defer_gpu_objects): create packets held now, written on first
    // use so far, and dropped at release without ever being written.
    uint64_t gpu_objects_pending = 0;
    uint64_t gpu_objects_created_on_use = 0;
    uint64_t gpu_objects_never_created = 0;
    // True when the submit lock stayed held for the whole bounded wait in stats() (a submitter
    // blocked in a full live pump, ...): the gpu_objects_* fields above are then 0, not read.
    bool submit_lock_busy = false;
};

class Resource;
class Heap;
class CommandList;
struct GraphicsPsoSource;

class Device final : public ObjectBase<IDeviceHost> {
public:
    static constexpr const char* kComClassName = "ID3D12Device";
    // Guest payload (after the vtable pointer, at device + 0x10): the byte at device + 0x18 is read
    // by inlined accessors in the executable (see the header comment).
    static constexpr uint64_t kComPayloadSize = 16;
    static constexpr uint64_t kInlineApiFlagOffset = 0x18;

    explicit Device(DeviceOptions options = {});
    ~Device() override;

    // Must run once right after create_object (writes the guest payload).
    void initialize_guest_payload();

    // ---- IDeviceHost ----
    uint32_t GetNodeCount() override { return 1; }
    HResult CreateCommandQueue(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) override;
    HResult CreateCommandAllocator(uint32_t type, const Guid& riid, OutPtr out) override;
    HResult CheckFeatureSupport(uint32_t feature, uint64_t data, uint32_t size) override;
    HResult CreateDescriptorHeap(const GuestDescriptorHeapDesc* desc, const Guid& riid, OutPtr out) override;
    uint32_t GetDescriptorHandleIncrementSize(uint32_t type) override;
    HResult CreateRootSignature(uint32_t node_mask, uint64_t blob, uint64_t length, const Guid& riid,
                                OutPtr out) override;
    void CreateConstantBufferView(uint64_t desc, uint64_t dest) override;
    void CreateSampler(uint64_t desc, uint64_t dest) override;
    void CopyDescriptors(com::CallContext& ctx) override;
    void CopyDescriptorsSimple(uint32_t count, uint64_t dest, uint64_t src, uint32_t type) override;
    GuestResourceAllocationInfo GetResourceAllocationInfo(uint32_t visible_mask, uint32_t count,
                                                          uint64_t descs) override;
    HeapPropertiesResult GetCustomHeapProperties(uint32_t node_mask, uint32_t heap_type) override;
    HResult MakeResident(uint32_t, uint64_t) override { return kSOk; }
    HResult Evict(uint32_t, uint64_t) override { return kSOk; }
    HResult CreateFence(uint64_t initial_value, uint32_t flags, const Guid& riid, OutPtr out) override;
    HResult GetDeviceRemovedReason() override { return kSOk; }
    void GetCopyableFootprints(com::CallContext& ctx) override;
    HResult CreateQueryHeap(const GuestQueryHeapDesc* desc, const Guid& riid, OutPtr out) override;
    HResult SetStablePowerState(bool) override { return kSOk; }
    HResult CreateCommandSignature(const GuestCommandSignatureDesc* desc, uint64_t root_signature,
                                   const Guid& riid, OutPtr out) override;
    HResult CreateCommandQueueX(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out) override;
    HResult CreateGraphicsPipelineState(uint64_t desc, const Guid& riid, OutPtr out) override;
    HResult CreateComputePipelineState(uint64_t desc, const Guid& riid, OutPtr out) override;
    HResult CreateCommandList(uint32_t node_mask, uint32_t type, InterfaceArg<ICommandAllocatorHost> allocator,
                              InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid, OutPtr out) override;
    void CreateShaderResourceView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) override;
    void CreateUnorderedAccessView(InterfaceArg<IResourceHost> resource, InterfaceArg<IResourceHost> counter,
                                   uint64_t desc, uint64_t dest) override;
    void CreateRenderTargetView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) override;
    void CreateDepthStencilView(InterfaceArg<IResourceHost> resource, uint64_t desc, uint64_t dest) override;
    HResult CreateCommittedResource(com::CallContext& ctx) override;
    HResult CreateHeap(const GuestHeapDesc* desc, const Guid& riid, OutPtr out) override;
    HResult CreatePlacedResource(com::CallContext& ctx) override;
    HResult CreateReservedResource(uint64_t desc, uint32_t state, uint64_t clear, const Guid& riid,
                                   OutPtr out) override;
    HResult CreateCommandListX(const GuestCommandListDescX* desc, InterfaceArg<ICommandAllocatorHost> allocator,
                               InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid, OutPtr out) override;
    HResult CreateCommittedResource3(com::CallContext& ctx) override;
    HResult CreatePlacedResourceX(com::CallContext& ctx) override;
    // Xbox slots (d3d12_xs_xbox.cpp).
    HResult Slot49_SetFeatureToggle_unverified(uint32_t feature, uint32_t value) override;
    HResult Slot50_RegisterRemovalCallbacks_unverified(uint64_t a, uint64_t b, uint64_t c) override;
    HResult CreatePipelineStateWithOverrides(InterfaceArg<IPipelineStateHost> base, uint32_t count, uint64_t overrides,
                                             const Guid& riid, OutPtr out) override;
    HResult CreateNativePipelineState(uint64_t desc, const Guid& riid, OutPtr out) override;
    HResult Slot61_unverified(uint64_t a, uint64_t b) override;
    HResult CreateComputePipelineStateX(com::CallContext& ctx) override;
    HResult CreatePipelineStateFromBlob_unverified(com::CallContext& ctx) override;
    void CreateShaderResourceViewAtAddress(uint64_t resource_desc, uint64_t view_desc, uint64_t dest) override;
    void CreateUnorderedAccessViewAtAddress(uint64_t resource_desc, uint64_t view_desc, uint64_t dest) override;
    HResult RegisterPagePool_unverified(uint64_t address, uint32_t pages, GuestPtr<uint64_t> handle) override;
    HResult SetFrameIntervalX(uint64_t output, uint32_t interval_us, uint32_t period, uint32_t flags) override;
    HResult ScheduleFrameEventX(uint32_t type, uint32_t offset_us, uint64_t reserved, uint32_t flags) override;
    HResult WaitFrameEventX(uint32_t type, uint32_t timeout_ms, uint64_t reserved, uint32_t flags,
                            GuestPtr<uint64_t> token) override;
    HResult QueryFrameStatistics_unverified(uint64_t token, uint32_t flags, GuestPtr<uint32_t> count,
                                            uint64_t entries) override;
    HResult CreatePipelineStateFromStream(uint64_t stream_desc, const Guid& riid, OutPtr out) override;
    HResult CreateStateObject(uint64_t desc, const Guid& riid, OutPtr out) override;
    GuestResourceAllocationInfo GetResourceAllocationInfo1(uint32_t visible_mask, uint32_t count, uint64_t descs,
                                                           uint64_t infos) override;
    HResult Slot119_unverified(uint64_t buffer, uint32_t size, uint32_t flags) override;
    HResult Slot132_unverified(uint32_t value) override;

    // ---- Xbox frame model (d3d12_xs_xbox.h) ----
    xbox::FramePacer& frame_pacer() { return pacer_; }
    // O36: the sync interval of a PresentX in 60 Hz vblanks: the frame interval the title set
    // (slot 78), rounded, at least one.
    uint32_t present_interval_vblanks() const {
        const uint32_t us = pacer_.interval_us();
        return std::max<uint32_t>(1, (us + 8333) / 16667);
    }
    // Wakes every WaitFrameEventX and makes later ones return at once (shutdown, tests).
    void shutdown_frame_events() {
        pacer_.shutdown();
        stop_frame_event_thread();
        {
            std::lock_guard<std::mutex> lock(present_mutex_);
            present_shutdown_ = true;
        }
        present_cv_.notify_all();
    }
    // PresentX of a frame token: on our side the frame's submissions (and their CPU-visible
    // buffer snapshots) are done. Wakes PRESENT waits (WaitFrameEventX type 2) for that token.
    void note_present_token(uint64_t token);
    // Blocks until the frame token has been presented (true), the timeout passes or the frame
    // events shut down (false). `timeout_ms` = kInfiniteFrameWait waits without a bound.
    bool wait_present_token(uint64_t token, uint32_t timeout_ms);
    static constexpr uint32_t kInfiniteFrameWait = 0xFFFFFFFFu;
    // Fence completion latency (DeviceOptions::fence_publish_lag): fences with pending queue
    // signals, the present epoch (PresentX count) and the publisher.
    bool fence_publish_lag() const { return options_.fence_publish_lag && options_.fence_publish_lag_frames > 0; }
    uint32_t fence_publish_lag_frames() const {
        return options_.fence_publish_lag ? options_.fence_publish_lag_frames : 0u;
    }
    uint32_t fence_publish_cap_ms() const { return options_.fence_publish_cap_ms; }
    uint64_t present_epoch() const { return present_epoch_.load(std::memory_order_acquire); }
    // GPU -> CPU read-back ordering (d3d12_readback.h).
    ReadbackGate& readbacks() { return *readback_gate_; }
    void register_lag_fence(Fence* fence);
    void unregister_lag_fence(Fence* fence);
    // Called at every PresentX (after the present): advances the epoch and publishes due values.
    void on_frame_presented();
    // Publishes the values the epoch rule or the cap admits on every registered fence.
    void publish_fences();
    void count(xbox::Counter c, uint64_t n = 1) { xbox_counters_[static_cast<size_t>(c)].fetch_add(n); }
    uint64_t counter(xbox::Counter c) const { return xbox_counters_[static_cast<size_t>(c)].load(); }
    // Last frame token passed to ExecuteCommandLists / a present plane (0 = none).
    std::atomic<uint64_t> last_execute_token{0};
    std::atomic<uint64_t> last_present_token{0};

    // ---- Pipeline-state creation from host-side descriptions (also used by the Xbox slots) ----
    HResult create_graphics_pso(const GuestGraphicsPipelineStateDesc& g, const Guid& riid, OutPtr out);
    HResult create_graphics_pso_from(GraphicsPsoSource source, const Guid& riid, OutPtr out);
    HResult create_compute_pso(const GuestComputePipelineStateDesc& g, const Guid& riid, OutPtr out);
    // A pipeline state that binds as "unresolved" (draws skipped) for pipelines the layer cannot
    // build (mesh shaders, ...): handed out instead of a null pointer the caller would dereference.
    HResult create_placeholder_pso(const std::string& reason, const Guid& riid, OutPtr out);
    // G66: a mesh pipeline (stream with an MS subobject) emulated as the mesh stage run as compute
    // plus a draw over its output (bridge::MeshPipelineState) when the shader pack has the stage's
    // compute translation (bridge::mesh_compute_pack_key) with its mesh_output reflection; a
    // placeholder otherwise (and for amplification shaders).
    HResult create_mesh_pso(const GuestGraphicsPipelineStateDesc& g, const GuestShaderBytecode& ms,
                            bool amplification, const Guid& riid, OutPtr out);
    // True when at least one Compute entry in the shader pack has a reflection entry with
    // mesh_output (the only stages create_mesh_pso can resolve). Computed once lazily.
    bool mesh_pipelines_possible() const;
    // A console-native graphics pipeline (Xbox slot 57, d3d12_xs_xbox.h): unresolved, with one
    // identity per stage (native_stage_identity).
    HResult create_native_pso(const xbox::NativePipeline& pipeline, const Guid& riid, OutPtr out);

    // ---- Shaders ----
    // Shader identity (shader_identity(): FNV-1a 64 over the DXIL container bytes the PSO desc passes)
    // -> CreateShaderModule id. PSOs created before their modules are registered stay unresolved.
    void register_shader_module(uint64_t identity, uint32_t module_id);
    enum class ShaderStage : uint8_t { Vertex, Pixel, Compute };
    void note_unresolved_shader(uint64_t identity, ShaderStage stage);
    std::vector<uint64_t> unresolved_shaders() const;
    // Distinct unresolved identities by the stage they were first used at (vertex, pixel, compute).
    std::array<uint32_t, 3> unresolved_shader_stages() const;
    // Root signatures created, by blob version (1.0, 1.1, 1.2).
    std::array<uint32_t, 3> root_signature_versions() const;
    // Console-native stages (Xbox slot 57): distinct identities are counted per stage (vertex,
    // pixel) instead of being logged one by one; the first native pipeline is logged once. They are
    // not in unresolved_shaders() (that list is the DXIL the shader track can translate).
    void note_native_stages(uint64_t vertex_identity, uint64_t pixel_identity);
    std::array<uint32_t, 2> native_stage_counts() const;
    // A console-native stage as the guest layer sees it (Xbox slot 57).
    struct NativeStageRecord {
        uint64_t identity = 0;       // native_stage_identity: code then metadata
        uint64_t code_identity = 0;  // FNV-1a 64 over the code range only
        uint32_t code_bytes = 0;
        uint32_t metadata_bytes = 0;
    };
    using DxilDigest = std::array<uint8_t, 16>;
    // Reads a stage's code and metadata (false: unreadable, diagnosed). With `index`, the stage is
    // also indexed by the 16-byte values at its metadata's digest positions (d3d12_xs_xbox.h,
    // "Native pipelines and their DXIL") so find_native_stage can join a DXIL container to it.
    // `pixel` / `layout_b` only label the stage in the pipeline dump.
    bool read_native_stage(const xbox::NativeStage& stage, bool index, NativeStageRecord* out, bool pixel = false,
                           bool layout_b = false);
    // The indexed native stage whose metadata carries `digest` (a DXIL container's HASH digest).
    bool find_native_stage(const DxilDigest& digest, NativeStageRecord* out) const;
    size_t native_index_size() const;
    // Bounded content-addressed cache of embedded native stages extracted from DXIL containers
    // (keyed by FNV-1a 64 shader identity and stage kind) so repeated PSOs skip re-inflating XDAT.
    bool find_cached_embedded_native(uint64_t identity, uint8_t want, NativeStageRecord* out) const;
    void cache_embedded_native(uint64_t identity, uint8_t want, const NativeStageRecord& record);
    // Pipeline dump (DeviceOptions::dump_dir), null when off. flush_pipeline_dump rewrites the
    // manifest (also done every 60 presents and when the device is destroyed).
    PipelineDump* pipeline_dump() const { return dump_.get(); }
    // A GPU-only buffer owned by the runtime (rectangle-list expansion); empty on failure.
    com::Ref<Resource> create_internal_buffer(uint64_t size);
    // Uniform bindings WebGPU cannot bind where they are (BridgeOptions::relocate_uniform): at an
    // unaligned offset, or smaller than the stage's uniform block with no room behind them in
    // their buffer. The bytes at [va, va + size) (CPU-visible memory only) get a 256-byte aligned
    // home of at least `binding_size` bytes in runtime-owned buffers, refreshed from guest memory
    // at every submission (flush_cpu_buffers) when changed. The home past `size` stays zero: that
    // is the zero-extension WebGPU needs to bind the declared block, not data (a console stage
    // reading past a small constant buffer reads whatever memory follows).
    bool relocate_uniform(uint64_t va, uint64_t size, uint64_t binding_size, uint32_t* buffer_id, uint64_t* offset);
    size_t relocated_uniform_count() const;
    // O36: textures known only by a view at an address (bounded; least recently used dropped).
    static constexpr size_t kMaxAddressTextures = 2048;
    // At the cap, this many least recently used are dropped in one pass (one victim scan, one
    // bind-group invalidation pass): a title whose working set exceeds the cap otherwise paid both
    // per new texture, seconds per frame in the 3D scene.
    static constexpr size_t kAddressTextureEvictBatch = 64;
    size_t address_texture_count() const;
    // Storage buffer views at unaligned offsets or address-only GPU VAs (G51): the bytes at
    // [va, va + size) (readable guest/alias memory only) get a 256-byte aligned home (or a
    // dedicated offset-0 buffer when `require_zero_offset` is true, e.g. element stride does not
    // divide 256) in runtime-owned Storage|CopyDst buffers registered with the state bridge,
    // uploaded immediately and refreshed on flush_cpu_buffers when guest bytes change.
    bool relocate_storage_buffer(uint64_t va, uint64_t size, uint32_t* buffer_id, uint64_t* offset,
                                 bool require_zero_offset = false);
    size_t relocated_storage_count() const;
    // Bounds of the storage relocation cache. Past kStorageRelocationSoftEntries a home unused for
    // three presents is handed to the new range; the cache grows otherwise, up to
    // kStorageRelocationHardEntries entries or kStorageRelocationHardBytes of homes, where the least
    // recently used home is evicted even if it is recent. A world frame binds thousands of unaligned
    // read-only storage views: a hard bound below that working set makes every bind a miss (each one
    // a fresh copy of up to 16 MiB, and each eviction forces every list that binds a home to resolve
    // its relocations again), which took 20 s per frame in the in-game load (stall after ~66 s).
    static constexpr size_t kStorageRelocationSoftEntries = 512;
    static constexpr size_t kStorageRelocationHardEntries = 65536;
    static constexpr uint64_t kStorageRelocationHardBytes = uint64_t{1} << 30;
    // Settled homes a miss looks at for reuse, oldest first, before the least recently used one is dropped.
    static constexpr size_t kStorageReuseScan = 1024;
    uint64_t relocated_storage_bytes() const;
    // Shared (non-dedicated) 4 MiB chunks allocated for small storage relocations (test hook, #2093).
    size_t relocated_storage_chunk_count() const;
    // #663: true for a buffer that holds storage relocation homes (BridgeOptions::relocation_home).
    bool storage_relocation_home(uint32_t buffer_id) const;
    // BridgeOptions::storage_homes_unmoved: true when every storage home in `homes` (buffer id,
    // offset) is still the live home it was when relocation_epoch() read `since_epoch` (not handed
    // to another range or destroyed since); each one is then marked used, as a relocate_storage_buffer
    // hit would. False when any of them moved (the caller resolves the bind group again).
    bool storage_homes_unmoved(const std::pair<uint32_t, uint64_t>* homes, size_t count, uint64_t since_epoch);
    // Index / vertex buffer views at address-only GPU VAs (G58: dynamic guest-RAM ring buffers
    // with no ID3D12Resource): the bytes at [va, va + size) (readable guest/alias memory only, at
    // most 16 MiB) get a dedicated Vertex|Index|CopyDst buffer at offset 0, uploaded immediately
    // and refreshed on flush_cpu_buffers when guest bytes change (FNV-1a 64). A request inside an
    // existing relocation's range (a ring buffer advancing its write pointer) reuses that buffer
    // at offset va - base when the offset is 4-byte aligned. Bounded LRU (entries and bytes).
    bool relocate_ia_buffer(uint64_t va, uint64_t size, uint32_t* buffer_id, uint64_t* offset);
    size_t relocated_ia_count() const;
    // Shader pack resolution (DeviceOptions::shader_pack). Each stage of a graphics pipeline is
    // looked up by the code-only identity of the console-native stage its DXIL joins to, else by
    // its DXIL identity. When every stage has a key, the stages get modules (one CreateShaderModule
    // per key, code = pack_key_code(key), resolved by the WebGPU side from its pack) and, unless
    // one is registered, a pipeline layout derived from the root signature (one group; bindings
    // numbered from 0 in root-parameter order: table ranges descriptor by descriptor, root
    // descriptors and root constants one each, then static samplers). Returns the keys used
    // (0 = stage absent); false when a stage has no key.
    bool resolve_with_pack(GraphicsPsoSource& source, uint64_t* vs_key, uint64_t* ps_key);

    // Native prewarm (pipeline_prewarm_host.h): creates the shader modules of a warm-list entry ahead of
    // the title's first use, through the same path as on-demand creation (the pack module cache).
    // Safe from any thread. False when there is no pack or it cannot resolve a key.
    bool prewarm_pack_modules(uint32_t kind, uint64_t key0, uint64_t key1);
    // Full pipeline of a warm-list entry (pipeline_prewarm.h, list version 2): its modules, layout and host
    // pipeline are created ahead of the title with the ids the title's own request for the same pipeline
    // would use (one build, under the lock a concurrent title request takes). Safe from any thread.
    bridge::PrewarmPipelineResult prewarm_pipeline(uint32_t task_id, uint32_t kind, uint64_t key0, uint64_t key1,
                                                   const std::vector<uint8_t>& detail);
    uint32_t pack_module_locked(uint64_t key, GpuShaderStage stage);  // pack_mutex_ held
    // The @group(0) renumbering of a pack stage (StageReflection::group0_binding_shift; 0 = none): its
    // modules are created as pack_key_code(key, shift) so they match the joined layouts.
    uint32_t pack_binding_shift(uint64_t key) const;
    // Module for a pack key (deterministic id, built once) or, for a non-pack key, a sequential id. pack_mutex_ held.
    uint32_t module_locked(uint64_t key, GpuShaderStage stage, const std::string& code);
    // Reflection join failures reported so far (once per pipeline: stage keys + root signature).
    size_t reflection_join_failures() const;
    void flush_pipeline_dump();
    bool shader_known(uint64_t identity) const;
    // Bytecode length of each shader identity seen in a pipeline state (0 if unknown).
    void note_shader_size(uint64_t identity, uint64_t bytes, int kind = -1);
    uint64_t shader_size(uint64_t identity) const;
    // The shader kind the DXIL container declares (program header; 0 pixel, 1 vertex, 5 compute,
    // ...), -1 if unknown.
    int shader_kind(uint64_t identity) const;
    // The first skipped draw of each pipeline state is described once (counts, formats, sizes):
    // claim_skipped_draw_report is true exactly once per pipeline state.
    bool claim_skipped_draw_report(const void* pipeline_state);
    void add_skipped_draw_report(const std::string& text);
    std::vector<std::string> skipped_draw_reports() const;

    // ---- Resources ----
    // GPU virtual address range for memory without guest backing (default heaps, reserved
    // resources): kFakeGpuVaBase upwards, 64 KiB aligned, never reused.
    static constexpr uint64_t kFakeGpuVaBase = 0x00007E0000000000ull;
    uint64_t allocate_gpu_va(uint64_t size);
    // CPU-visible buffers whose guest memory is compared with a shadow copy at every
    // ExecuteCommandLists (changed 4 KiB pages become WriteBuffer packets).
    void track_cpu_buffer(Resource* resource);
    void untrack_cpu_buffer(Resource* resource);
    void flush_cpu_buffers(CommandEncoder& encoder);
    // Dirty-tracked guest texture memory upload (G49b, G64): tracks FNV-1a 64 hashes of the whole
    // source surface and of each subresource, plus layout metadata, for guest-memory-backed GPU
    // textures, and emits one WriteTexture per changed subresource (every mip and array slice of a
    // console-tiled surface) when bound SRV memory changes.
    void track_guest_texture(const TrackedTexture& tex);
    void untrack_guest_texture(uint32_t resource_id);
    bool tracked_texture_info(uint32_t resource_id, TrackedTexture* out) const;
    void set_detile_fn(DetileFn fn);
    bool sync_bound_texture_memory(uint32_t resource_id, CommandEncoder& encoder);
    // Brackets one ExecuteCommandLists on the calling thread: inside it each tracked texture is
    // synced at most once (TrackedTexture::synced_submission).
    void begin_submission();
    void end_submission();
    // Marks a texture as the destination of a recorded copy (CopyTextureRegion). The copy writes
    // texels in API coordinates and defines the contents from then on; the guest memory at the
    // texture's address is no longer uploaded (or de-tiled) over it.
    void note_texture_copy_destination(uint32_t resource_id);
    // Compute read-back (backlog 8): the texture's guest memory follows the GPU, so uploads stop,
    // but only while the guest bytes are what our own re-tile wrote. _pending marks the texture
    // (no upload until the first re-tile lands); _retiled records the surface hash right after the
    // host write. A later sync that finds different bytes saw a genuine guest CPU write: it drops
    // the ownership and uploads again.
    void note_texture_readback_pending(uint32_t resource_id);
    void note_texture_readback_retiled(uint32_t resource_id, uint64_t cpu_va, size_t bytes);
    // Creates (or returns a cached) WebGPU texture for a GPU VA that has no ID3D12Resource
    // (Xbox Slot-70/71 address views and raw GFX10 T# image descriptors in descriptor memory).
    // Registers TextureInfo in the state bridge and tracks guest memory for detiled WriteTexture
    // uploads on bind. Returns 0 when dimensions or format are unsupported.
    uint32_t ensure_address_texture(uint64_t gpu_va, uint32_t dxgi_format, uint32_t width, uint32_t height,
                                    uint32_t depth, uint32_t mip_levels, uint32_t d3d12_layout,
                                    GpuTextureDimension dim, bool uav);
    // Designates a texture as the presented back buffer: render passes that target it write the
    // canvas texture (kGpuSwapchainTextureId). For hosts that own the swapchain; PresentX does not
    // call it (it draws the presented texture into the canvas instead, Device::present). False if
    // `resource` is not a 2D texture.
    bool set_back_buffer(uint64_t resource_interface_pointer);
    // GPU address windows aliasing CPU memory (the executable reserves a CPU window and a GPU window
    // of equal size and maps the same pages into both; d3d12_xs_xbox.h). A buffer created at a GPU
    // address inside a registered window reads its contents from the matching CPU address.
    void register_gpu_alias(uint64_t gpu_base, uint64_t cpu_base, uint64_t size);
    uint64_t cpu_address_for_gpu(uint64_t gpu_va) const;  // gpu_va itself outside every window
    void note_upload(uint64_t bytes) { upload_bytes_.fetch_add(bytes, std::memory_order_relaxed); }
    void note_shadow_bytes(int64_t covered, int64_t resident) {
        shadow_bytes_.fetch_add(static_cast<uint64_t>(covered), std::memory_order_relaxed);
        shadow_resident_bytes_.fetch_add(static_cast<uint64_t>(resident), std::memory_order_relaxed);
    }
    void note_shadow_compared(uint64_t bytes) { shadow_compared_bytes_.fetch_add(bytes, std::memory_order_relaxed); }
    void note_texture_upload(uint64_t count = 1) { textures_uploaded_.fetch_add(count, std::memory_order_relaxed); }
    void note_skipped_draw() { skipped_draws_.fetch_add(1, std::memory_order_relaxed); }
    void note_unsupported_copy(const std::string& message);
    // True when a texture of `dxgi_format` gets a substitute WebGPU format whose texels differ from
    // the guest's (BC without texture-compression-bc, packed 16-bit RGBA, promoted RT formats): the
    // guest's bytes cannot be uploaded as they are, and no CPU conversion exists yet, so uploads to
    // it are skipped.
    bool upload_needs_conversion(uint32_t dxgi_format,
                                 pso::FormatUsage usage = pso::FormatUsage::Sampled) const;

    // ---- Descriptor words in descriptor memory (bridge/descriptor_words.h, G48) ----
    // After a registry write at `cpu_handle`: writes the words the console's D3D12 leaves in the
    // slot (heaps without guest memory are skipped). The title copies whole slots itself when it
    // fills shader-visible tables, so the words must be in its memory.
    void store_descriptor_words(uint64_t cpu_handle);
    // Writes `words` (one slot of the heap's increment) at `cpu_handle` (Xbox slots 70 / 71).
    void store_descriptor_words(uint64_t cpu_handle, const uint32_t* words, size_t bytes);
    // CopyDescriptors*: the console copies the slots' memory along with the descriptors.
    void copy_descriptor_memory(const std::vector<std::pair<uint64_t, uint32_t>>& dst_ranges,
                                const std::vector<std::pair<uint64_t, uint32_t>>& src_ranges, uint32_t type);
    // Slots written per outcome (index = bridge::WordsStatus).
    std::array<uint64_t, 7> descriptor_word_counts() const;
    // Root-table slots read back from descriptor memory, per binding kind (0 CBV, 1 SRV, 2 UAV,
    // 3 sampler) and outcome (index = bridge::DecodeStatus). Ok slots bind; every other outcome binds
    // null and its first occurrence per kind is logged with the reason.
    using SlotReadCounts = std::array<std::array<uint64_t, 7>, 4>;
    SlotReadCounts descriptor_slot_reads() const;
    // Xbox slots 70 / 71 after parsing the view description (d3d12_xs_xbox.cpp).
    void view_at_address(bool uav, uint64_t resource_desc, const d3d::ViewDesc& desc, uint64_t va, uint64_t dest);
    uint64_t count_texture_view_at_address();

    // ---- Runtime side ----
    d3d::DescriptorHeapRegistry& heaps() { return heaps_; }
    d3d::ResourceStateRegistry& resource_states() { return resource_states_; }
    bridge::StateBridgeDevice& state_bridge() { return *bridge_; }
    uint32_t allocate_id() { return next_id_.fetch_add(1, std::memory_order_relaxed); }
    uint64_t timestamp_frequency() const { return options_.timestamp_frequency; }

    // Writes `bytes` to the sink as one submission (thread-safe, serialized).
    void submit(const uint8_t* data, size_t size);
    void submit(const CommandEncoder& encoder) { submit(encoder.data(), encoder.size()); }
    // Several byte ranges as one submission (no other submission in between).
    void submit_parts(const std::vector<std::pair<const uint8_t*, size_t>>& parts);
    // Queues `encoder`'s packets for the start of the next submission (submit_parts / present) instead of
    // writing them now: never blocks on the sink (the live pump's backpressure), so it may be called
    // with a device lock held. Every submission that can reference what the packets create or write
    // (a command list binding a new storage home) is emitted after them.
    void submit_with_next(const CommandEncoder& encoder);
    // presents_ without stats()'s bounded wait for the submit lock (hot paths).
    uint64_t presents() const { return presents_.load(std::memory_order_relaxed); }
    // Present packet (and BeginFrame before the next frame's first submission). `plane_textures`
    // (PresentX planes, plane 0 on top; 0 = no texture) are first drawn into the canvas texture
    // (kGpuSwapchainTextureId) by one pass, each scaled to the full canvas: the last drawable plane
    // opaque, the ones above it blended as premultiplied alpha. So every presented back buffer
    // reaches the screen whenever its render-target views were created. Planes that cannot be
    // sampled with filtering, or that already are the canvas (set_back_buffer), are skipped.
    // Returns the number of planes drawn.
    uint32_t present(std::span<const uint32_t> plane_textures = {});
    void set_sink(PacketSink sink);
    // G79: Emits a WebGPU-valid MSAA resolve pass (4x MSAA 2D/2D-array subresource -> 1x 2D/2D-array
    // subresource) into `enc` using ABI v2 render-pass packets. Returns false when the format or
    // geometry cannot be resolved on WebGPU so the caller can fall back cleanly.
    bool emit_msaa_resolve(uint32_t dst_texture_id, uint32_t dst_mip, uint32_t dst_layer,
                           uint32_t src_texture_id, uint32_t src_layer, uint32_t resolve_dxgi_format,
                           CommandEncoder& enc);

    // ---- Lazy GPU objects (G66, #1077; DeviceOptions::defer_gpu_objects) ----
    // `create` holds the create packet(s) of resource `id`: it is written right before the first
    // submitted packet that references `id` (submit, submit_parts and present all check), or at once
    // when deferral is off.
    void defer_create(uint32_t id, const CommandEncoder& create);
    // Holds a device-global bridge object's creation packet(s) (layouts, samplers, pipelines,
    // runtime helper buffers/textures/bind groups) until the first submitted packet references `id`.
    void defer_device_object(uint32_t id, const CommandEncoder& create);
    // Re-creates texture `p.id` (the id's create packet replaces the texture): rewrites its held create
    // when that was never written, else appends a CreateTexture2 to `encoder` (stream order).
    void replace_host_texture(const GpuCreateTexture2Payload& p, CommandEncoder& encoder);
    // Resource `id` is released: true if its create packet was never written (it is dropped; the
    // GPU object never existed, so no DestroyResource must be sent).
    bool release_deferred(uint32_t id);
    // True while `id`'s create packet is held.
    bool gpu_object_pending(uint32_t id) const;

    // Never blocks for long: the fields guarded by the submit lock are read only when the lock is
    // free within kStatsLockWait (fault/stop reports call this while a submitter may sit in a full
    // live pump holding it); presents/submissions are atomics and always read.
    static constexpr std::chrono::milliseconds kStatsLockWait{100};
    DeviceStats stats() const;
    void note_bad_argument(const std::string& message);
    // CheckFeatureSupport ids the device does not answer (E_NOTIMPL): id -> times asked; logged once each.
    void note_unanswered_feature(uint32_t feature);
    std::map<uint32_t, uint64_t> unanswered_features() const;
    std::vector<std::string> diagnostics() const;

    // Child counters.
    std::atomic<uint64_t> queues{0}, allocators{0}, descriptor_heaps{0}, root_signatures{0}, fences{0};
    std::atomic<uint64_t> resources{0}, memory_heaps{0}, pipeline_states{0}, command_lists{0};

private:
    HResult create_queue(const GuestCommandQueueDesc* desc, const Guid& riid, OutPtr out);

    DeviceOptions options_;
    d3d::DescriptorHeapRegistry heaps_;
    d3d::ResourceStateRegistry resource_states_;
    std::atomic<uint32_t> next_id_{1};
    std::unique_ptr<bridge::StateBridgeDevice> bridge_;

    mutable std::mutex submit_mutex_;
    PacketSink sink_;
    std::unique_ptr<ReadbackGate> readback_gate_;
    // With submit_mutex_ held: `bytes` (whole packets) to the sink, each held create packet written
    // before the first packet that references its id (emit_locked), or as they are (write_sink_locked).
    void emit_locked(const uint8_t* bytes, size_t n);
    void write_sink_locked(const uint8_t* bytes, size_t n);
    std::unordered_map<uint32_t, std::vector<uint8_t>> deferred_creates_;         // under submit_mutex_
    std::unordered_map<uint32_t, std::vector<uint8_t>> deferred_device_objects_;  // under submit_mutex_
    uint64_t gpu_objects_created_on_use_ = 0;                              // under submit_mutex_
    uint64_t gpu_objects_never_created_ = 0;                               // under submit_mutex_
    bool frame_open_ = false;
    std::atomic<uint64_t> presents_{0};     // written under submit_mutex_, read lock-free by stats()
    std::atomic<uint64_t> submissions_{0};  // written under submit_mutex_, read lock-free by stats()
    // submit_with_next: packets for the start of the next submission. A leaf lock (taken under
    // submit_mutex_ or a device lock, never the other way round).
    std::mutex next_submission_mutex_;
    std::vector<uint8_t> next_submission_;
    void emit_next_submission_locked();  // with submit_mutex_ held
    // Present composition objects (under submit_mutex_), created on first use (PresentBlitCache).
    uint32_t emit_present_planes(std::span<const uint32_t> plane_textures, CommandEncoder& enc);
    PresentBlitCache present_blit_;
    // G79: MSAA resolve pipelines (color float/unorm/srgb & depth), created lazily once per device.
    mutable std::mutex resolve_mutex_;
    struct MsaaResolve {
        uint32_t vs = 0;
        uint32_t color_fs = 0, color_group_layout = 0, color_pipeline_layout = 0;
        uint32_t depth_fs = 0, depth_group_layout = 0, depth_pipeline_layout = 0;
        std::unordered_map<uint8_t, uint32_t> color_pipelines;  // WgpuTextureFormat code -> pipeline
        std::unordered_map<uint8_t, uint32_t> depth_pipelines;  // WgpuTextureFormat code -> pipeline
    } msaa_resolve_;

    mutable std::mutex diag_mutex_;
    std::vector<std::string> diags_;
    LogRateLimiter bad_arg_log_;  // guarded by diag_mutex_
    std::map<uint32_t, uint64_t> unanswered_features_;  // guarded by diag_mutex_
    std::atomic<uint64_t> bad_arguments_{0};

    HResult create_resource(const GuestResourceDesc& desc, const GuestHeapProperties& heap, uint32_t heap_flags,
                            uint32_t initial_state, com::Ref<Heap> placed_heap, uint64_t heap_offset,
                            uint64_t address, bool reserved, const Guid& riid, OutPtr out);
    HResult create_command_list(uint32_t type, InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid,
                                OutPtr out);

    mutable std::mutex shader_mutex_;
    std::vector<uint64_t> unresolved_shaders_;
    std::array<uint32_t, 3> unresolved_by_stage_{};
    std::array<uint32_t, 3> root_signature_versions_{};  // under shader_mutex_
    std::unordered_set<uint64_t> known_shaders_;
    std::unordered_set<uint64_t> native_stages_[2];  // under shader_mutex_ (vertex, pixel)
    struct DigestHash {
        size_t operator()(const DxilDigest& d) const {
            uint64_t a = 0, b = 0;
            std::memcpy(&a, d.data(), 8);
            std::memcpy(&b, d.data() + 8, 8);
            return static_cast<size_t>(a ^ (b * 0x9E3779B97F4A7C15ull));
        }
    };
    std::unordered_map<DxilDigest, NativeStageRecord, DigestHash> native_by_digest_;  // under shader_mutex_
    std::unordered_set<uint64_t> indexed_native_;                                     // under shader_mutex_
    struct EmbeddedCacheKey {
        uint64_t identity = 0;
        uint8_t want = 0;
        bool operator==(const EmbeddedCacheKey& o) const noexcept { return identity == o.identity && want == o.want; }
    };
    struct EmbeddedCacheKeyHash {
        size_t operator()(const EmbeddedCacheKey& k) const noexcept {
            return static_cast<size_t>(k.identity ^ (uint64_t{k.want} * 0x9E3779B97F4A7C15ull));
        }
    };
    static constexpr size_t kMaxEmbeddedNativeCache = 4096;
    std::unordered_map<EmbeddedCacheKey, NativeStageRecord, EmbeddedCacheKeyHash>
        embedded_native_cache_;  // under shader_mutex_
    mutable std::once_flag mesh_possible_once_;
    mutable bool mesh_possible_ = false;
    std::unique_ptr<PipelineDump> dump_;
    // (va, size, home bytes)
    using RelocatedUniformKey = std::tuple<uint64_t, uint64_t, uint64_t>;
    struct RelocatedUniform {
        uint64_t va = 0;
        uint64_t size = 0;  // guest bytes copied; the home is align_up(max(size, binding), 256), zero past them
        uint32_t buffer_id = 0;
        uint64_t offset = 0;
        std::vector<uint8_t> shadow;  // bytes last written (empty: never written)
        std::list<RelocatedUniformKey>::iterator lru_pos;  // this entry's place in relocation_lru_
        // #1963 page index (under relocation_mutex_): primed = compared after a successful flush
        // take covering its host range, so only compared again when a take reports it written.
        bool primed = false;
        uint64_t visit_seq = 0;
    };
    mutable std::mutex relocation_mutex_;
    // O36: homes of evicted entries, reused by the next entry of the same (size, home bytes) so the
    // zero tail past `size` stays zero. Bounds relocations_ and the chunks behind it.
    std::map<std::pair<uint64_t, uint64_t>, std::vector<std::pair<uint32_t, uint64_t>>> relocation_free_;
    std::map<RelocatedUniformKey, RelocatedUniform> relocations_;
    // relocations_ keys, least recently bound first (under relocation_mutex_): a bind moves its key to
    // the back, so the entry evicted at the cap is the front one (O(1) instead of a scan of the map).
    std::list<RelocatedUniformKey> relocation_lru_;
    struct RelocatedStorage;
    using StorageLru = StorageReuseLru<RelocatedStorage>;
    struct RelocatedStorage {
        uint64_t va = 0;
        uint64_t size = 0;
        uint32_t buffer_id = 0;
        uint64_t offset = 0;
        uint64_t slot_capacity = 0;
        uint64_t last_hash = 0;
        StorageLru::Node lru_node;  // in storage_relocation_lru_ (bind present, home shape)
        uint64_t created_epoch = 0;      // relocation_evictions_ when this home took the range
        bool ever_uploaded = false;
        bool dedicated = false;
        // Placed from bytes read while a flush ran (relocate_storage_buffer reads outside the lock): the
        // write marks cannot vouch for it, so the next flush hashes it whatever the marks say.
        bool hash_next_flush = false;
        // Native write tracking of the guest range: with a primed cursor and no page written since the last
        // flush the range is not hashed again (flush_cpu_buffers). Unprimed = hash and arm.
        os::WriteCursor write_cursor;
        bool write_primed = false;
        // Page index (storage_page_index_, as RelocatedUniform): compared after a successful flush take
        // covering its host range, so an indexed flush only looks at it again when a take reports it written.
        bool index_primed = false;
        uint64_t visit_seq = 0;
    };
    struct RelocatedStorageKey {
        uint64_t va = 0;
        uint64_t size = 0;
        bool dedicated = false;
        bool operator<(const RelocatedStorageKey& o) const noexcept {
            if (va != o.va) return va < o.va;
            if (size != o.size) return size < o.size;
            return dedicated < o.dedicated;
        }
    };
    mutable std::mutex storage_relocation_mutex_;
    std::map<RelocatedStorageKey, RelocatedStorage> storage_relocations_;
    // Host pages -> relocated storage ranges (under storage_relocation_mutex_), for indexed native
    // flushes; and the skipped entries that uploaded anyway under --flush-crosscheck.
    HostPageIndex storage_page_index_;
    std::unordered_set<const void*> crosscheck_storage_;
    // The storage_relocations_ entries with index_primed == false: with the ones a take reported, the
    // only storage ranges an indexed flush visits (as uniform_unprimed_).
    std::unordered_set<RelocatedStorage*> storage_unprimed_;
    // Live homes by (buffer id << 32 | offset), for storage_homes_unmoved.
    std::unordered_map<uint64_t, std::map<RelocatedStorageKey, RelocatedStorage>::iterator> storage_home_index_;
    static uint64_t storage_home_key(uint32_t buffer_id, uint64_t offset) noexcept {
        return (uint64_t{buffer_id} << 32) | (offset & 0xffffffffull);
    }
    std::vector<uint32_t> storage_relocation_buffers_;
    // The ids in storage_relocation_buffers_, under their own lock: storage_relocation_home is asked
    // per bound buffer while recording and must not wait behind flushes or relocation misses.
    mutable std::mutex storage_home_ids_mutex_;
    std::unordered_set<uint32_t> storage_home_ids_;
    void note_storage_home_buffer(uint32_t buffer_id, bool live);
    // Slots of evicted small (non-dedicated) homes inside the shared chunks, by capacity: a new small
    // home takes one (capacity in [need, 2 * need]) before the chunk bump pointer advances, so evicting
    // and re-creating relocations does not grow the chunks without bound (#2093).
    std::multimap<uint64_t, std::pair<uint32_t, uint64_t>> storage_relocation_free_;
    std::map<uint32_t, uint32_t> storage_relocation_chunk_live_;  // shared chunk -> live homes; 0 = destroyed
    uint32_t storage_relocation_buffer_ = 0;
    uint64_t storage_relocation_used_ = 0;
    // storage_relocations_ entries, least recently bound first: a bind moves its node to the back (O(1),
    // no allocation), eviction takes the front, and the miss path's reuse lookup is indexed (the same
    // home as a scan of the first kStorageReuseScan nodes). And the bytes of their homes (slot capacities).
    StorageLru storage_relocation_lru_{kStorageReuseScan};
    // A bind of `r` (under storage_relocation_mutex_): its node goes to the back with this present.
    void touch_storage_lru(RelocatedStorage& r, uint64_t present);
    uint64_t storage_relocation_bytes_ = 0;
    // G66 (#691): uniform / storage relocation homes handed to another range (BridgeOptions::relocation_epoch).
    std::atomic<uint64_t> relocation_evictions_{0};
    struct RelocatedIa {
        uint64_t va = 0;
        uint64_t size = 0;
        uint64_t buffer_bytes = 0;   // WebGPU buffer size (guest size rounded up to 4)
        uint32_t buffer_id = 0;
        uint64_t last_hash = 0;
        uint64_t last_used_seq = 0;
        os::WriteCursor write_cursor;  // as RelocatedStorage
        bool write_primed = false;
        // Page index (ia_page_index_, as RelocatedStorage): compared after a successful flush take
        // covering its host range, so an indexed flush only looks at it again when a take reports it written.
        bool index_primed = false;
        uint64_t visit_seq = 0;
    };
    mutable std::mutex ia_relocation_mutex_;
    std::map<std::pair<uint64_t, uint64_t>, RelocatedIa> ia_relocations_;  // (base GPU VA, size)
    // Host pages -> relocated IA ranges (under ia_relocation_mutex_), for indexed native flushes;
    // and the skipped entries that uploaded anyway under --flush-crosscheck.
    HostPageIndex ia_page_index_;
    std::unordered_set<const void*> crosscheck_ia_;
    // The ia_relocations_ entries with index_primed == false: with the ones a take reported, the
    // only IA ranges an indexed flush visits (as uniform_unprimed_ / storage_unprimed_).
    std::unordered_set<RelocatedIa*> ia_unprimed_;
    uint64_t ia_relocation_bytes_ = 0;
    uint64_t ia_relocation_seq_ = 0;
    mutable std::mutex pack_mutex_;
    std::unordered_map<uint64_t, uint32_t> pack_modules_;                 // non-pack key -> module id (sequential ids)
    std::unique_ptr<PackModuleTable> pack_table_;                         // pack key -> deterministic module id
    std::set<std::pair<uint64_t, uint64_t>> pack_layouts_;                // (vs, ps) identities
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> reflection_failures_;  // (vs key, ps key, root signature)
    uint32_t relocation_buffer_ = 0;  // current GPU buffer id (CreateBuffer packet only, no COM object)
    uint64_t relocation_used_ = 0;    // bytes used in relocation_buffer_
    std::atomic<uint64_t> dump_presents_{0};
    struct ShaderFacts {
        uint64_t bytes = 0;
        int kind = -1;
    };
    std::unordered_map<uint64_t, ShaderFacts> shader_sizes_;  // under shader_mutex_
    std::unordered_set<const void*> described_pipelines_;   // under shader_mutex_
    std::vector<std::string> skipped_draw_reports_;          // under shader_mutex_ (first 64)
    std::mutex va_mutex_;
    uint64_t next_fake_va_ = kFakeGpuVaBase;
    std::mutex tracked_mutex_;
    std::vector<Resource*> tracked_;
    os::guest_window::SlotIndex slot_index_;  // O36: window slot -> tracked buffers (under tracked_mutex_)
    // #1963: host pages -> tracked buffers (under tracked_mutex_) and relocated uniform ranges (under
    // relocation_mutex_), for native flushes driven by the OS write tracking (flush_cpu_buffers).
    HostPageIndex buffer_page_index_;
    HostPageIndex uniform_page_index_;
    // --flush-crosscheck: skipped entries that uploaded anyway (each must be reported at the next flush).
    std::unordered_set<const void*> crosscheck_buffers_;   // under tracked_mutex_
    // The tracked_ entries with flush_primed == false (under tracked_mutex_): with the ones a take
    // reported, the only buffers an indexed flush visits (as uniform_unprimed_ / storage_unprimed_).
    std::unordered_set<Resource*> tracked_unprimed_;
    std::unordered_set<const void*> crosscheck_uniforms_;  // under relocation_mutex_
    // The relocations_ entries with primed == false (under relocation_mutex_): with the ones a take
    // reported, the only uniform ranges an indexed flush visits.
    std::unordered_set<RelocatedUniform*> uniform_unprimed_;
    // Under flush_mutex_: the flush take's position, the flush sequence and whether the index is in use.
    os::WriteCursor flush_cursor_{};
    uint64_t flush_seq_ = 0;
    bool page_index_live_ = false;
    // Bumped when a flush begins and when it ends (odd while one runs; read without flush_mutex_):
    // relocate_storage_buffer reads a new range's bytes outside its lock and needs to know whether a
    // flush could have taken the write marks meanwhile.
    std::atomic<uint64_t> flush_epoch_{0};
    // O36: one pass over the translated window's write marks per flush_cpu_buffers, shared by the
    // tracked buffers and the relocated ranges (under flush_mutex_).
    std::mutex flush_mutex_;
    os::guest_window::MarkPass mark_pass_;
    // Reusable scratch buffers for flush_cpu_buffers (under flush_mutex_) to avoid per-submission
    // heap allocations in steady state.
    std::vector<uint64_t> flush_marked_;
    std::vector<uint64_t> flush_native_written_;
    std::vector<uint64_t> flush_dirty_slots_;
    std::vector<HostPageIndex::Range> flush_take_ranges_;
    std::vector<HostPageIndex::Range> flush_taken_ok_;
    std::vector<HostPageIndex::Range> flush_taken_failed_;
    std::vector<uintptr_t> flush_dirty_pages_;
    std::vector<std::pair<uintptr_t, uintptr_t>> flush_host_ranges_;
    std::vector<uint8_t> flush_take_ok_flags_;
    std::vector<const void*> flush_visit_keys_;
    std::vector<Resource*> flush_buffer_candidates_;
    std::vector<RelocatedUniform*> flush_uniform_candidates_;
    std::vector<RelocatedStorage*> flush_storage_candidates_;
    // A relocated storage range an indexed flush hashes without storage_relocation_mutex_: taken under
    // the lock (write marks, cursor), hashed after letting go (and, when changed, encoded straight from
    // the guest view into the flush's encoder: one copy), applied under it again if the entry is still
    // the same one (key, address and created_epoch); a stale entry's packets are patched to Nops.
    struct StorageHashJob {
        RelocatedStorage* entry = nullptr;
        uint64_t va = 0, size = 0;
        bool dedicated = false;
        uint64_t created_epoch = 0;
        uint64_t cpu = 0;
        uintptr_t taken_host = 0;  // host address the cursor was taken at (0: not mapped in one piece)
        bool ever_uploaded = false;
        uint64_t last_hash = 0;
        // Filled without the lock.
        uintptr_t host = 0;
        bool read_ok = false;
        bool changed = false;
        uint64_t hash = 0;
        std::vector<uint8_t> bytes;  // the bytes read when the range is not mapped in one piece
        uint32_t buffer_id = 0;      // the entry's home (fixed for an entry with this created_epoch)
        uint64_t offset = 0;
        size_t packets_begin = 0, packets_end = 0;  // its WriteBuffer packets in the flush encoder
        // Filled when applied.
        bool upload = false;
    };
    std::vector<StorageHashJob> flush_storage_jobs_;  // reused; the first flush_storage_job_count_ are live
    size_t flush_storage_job_count_ = 0;
    std::vector<RelocatedIa*> flush_ia_candidates_;
    std::vector<uint8_t> flush_bytes_scratch_;
    mutable std::mutex tracked_textures_mutex_;
    std::unordered_map<uint32_t, TrackedTexture> tracked_textures_;
    os::guest_window::SlotIndex texture_slot_index_;    // G68: window slot -> resource_id (under tracked_textures_mutex_)
    std::unordered_set<uint32_t> copy_destinations_;  // under tracked_textures_mutex_
    struct ReadbackOwned {
        bool valid = false;  // a re-tile has landed and `hash` is its surface hash
        uint64_t va = 0;
        size_t bytes = 0;
        uint64_t hash = 0;
        // Write tracking position taken before `hash` was last computed or confirmed: a bind that finds
        // no page written since then skips the surface hash. primed = that take could tell.
        os::WriteCursor cursor{};
        bool cursor_primed = false;
    };
    std::unordered_map<uint32_t, ReadbackOwned> readback_owned_;  // under tracked_textures_mutex_
    std::unordered_set<uint32_t> conversion_skipped_;  // under tracked_textures_mutex_: reported once
    std::unordered_set<uint32_t> lossy_reported_;      // under tracked_textures_mutex_: DXGI formats
    struct AddressTextureKey {
        uint64_t gpu_va = 0;
        uint32_t dxgi_format = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t depth = 1;
        uint32_t mip_levels = 1;
        uint32_t d3d12_layout = 0;
        uint8_t dim = 0;

        bool operator==(const AddressTextureKey& o) const noexcept {
            return gpu_va == o.gpu_va && dxgi_format == o.dxgi_format && width == o.width && height == o.height &&
                   depth == o.depth && mip_levels == o.mip_levels && d3d12_layout == o.d3d12_layout && dim == o.dim;
        }
    };
    struct AddressTextureKeyHash {
        size_t operator()(const AddressTextureKey& k) const noexcept {
            uint64_t h = uint64_t{0xcbf29ce484222325ull};
            auto mix = [&](uint64_t v) {
                h ^= v;
                h *= uint64_t{0x00000100000001b3ull};
            };
            mix(k.gpu_va);
            mix(k.dxgi_format);
            mix(k.width);
            mix(k.height);
            mix(k.depth);
            mix(k.mip_levels);
            mix(k.d3d12_layout);
            mix(k.dim);
            return static_cast<size_t>(h);
        }
    };
    mutable std::mutex address_textures_mutex_;
    struct AddressTextureEntry {
        uint32_t id = 0;
        uint64_t last_use = 0;
        uint64_t budget_bytes = 0;  // counted in the texture budget (gpu/texture_budget.h); 0 = not counted
        SmallTextureShape shape;    // the small texture pool key (gpu/small_texture_pool.h)
    };
    std::unordered_map<AddressTextureKey, AddressTextureEntry, AddressTextureKeyHash> address_textures_;
    uint64_t address_texture_tick_ = 0;  // under address_textures_mutex_
    // Key hashes of address textures dropped by the LRU (under address_textures_mutex_; cleared when
    // large): a later miss on one of them counts as a re-creation (small_texture_pool stats).
    std::unordered_set<uint64_t> evicted_address_keys_;
    static constexpr size_t kMaxEvictedAddressKeys = 1u << 16;
    void note_address_texture_evicted(const AddressTextureKey& key);  // under address_textures_mutex_
    mutable std::mutex detile_mutex_;
    DetileFn detile_texture_;
    bool using_default_detile_ = false;
    std::atomic<uint64_t> upload_bytes_{0};
    std::atomic<uint64_t> shadow_bytes_{0};
    std::atomic<uint64_t> shadow_resident_bytes_{0};
    std::atomic<uint64_t> shadow_compared_bytes_{0};
    std::atomic<uint64_t> textures_uploaded_{0};
    std::array<std::atomic<uint64_t>, 7> descriptor_words_{};
    std::atomic<uint32_t> descriptor_words_reported_{0};  // bit per WordsStatus already logged
    // Root tables read from descriptor memory (BridgeOptions::slot_source).
    class MemorySlotSource final : public d3d::DescriptorSlotSource {
    public:
        explicit MemorySlotSource(Device& device) : device_(device) {}
        bool reads(const d3d::DescriptorHeap& heap) const override;
        bool snapshot(const d3d::DescriptorHeap& heap, uint32_t first, uint32_t count,
                      std::vector<uint8_t>* out) const override;
        d3d::ViewDescriptor view(const d3d::DescriptorHeap& heap, uint32_t index,
                                 d3d::DescriptorRangeType range) const override;
        bool sampler(const d3d::DescriptorHeap& heap, uint32_t index, d3d::SamplerDesc* out) const override;
        // Constant-buffer contents for CB-indexed heap bindings (G62): guest memory at the CPU
        // address the GPU address aliases (cpu_address_for_gpu).
        bool read_gpu_memory(uint64_t gpu_address, void* out, size_t bytes) const override;

    private:
        void note(uint32_t kind, uint32_t status, uint64_t address) const;
        Device& device_;
    };
    std::unique_ptr<MemorySlotSource> slot_source_;
    mutable std::array<std::array<std::atomic<uint64_t>, 7>, 4> slot_reads_{};
    mutable std::atomic<uint32_t> slot_reads_reported_{0};  // bit per (kind, status) already logged
    std::atomic<uint64_t> skipped_draws_{0};
    std::atomic<uint64_t> unsupported_copies_{0};
    std::mutex present_mutex_;
    std::condition_variable present_cv_;
    bool present_shutdown_ = false;
    std::atomic<uint64_t> present_epoch_{0};
    std::mutex lag_mutex_;
    std::set<Fence*> lag_fences_;
    std::condition_variable lag_cv_;
    bool lag_stop_ = false;
    std::thread lag_publisher_;  // publishes capped values while no present comes
    xbox::FramePacer pacer_;
    // ScheduleFrameEventX with an event handle: signalled at every pacer tick, `offset_us` into it
    // (one entry per event type; scheduling a type again replaces it).
    struct ScheduledFrameEvent {
        uint32_t offset_us = 0;
        uint64_t handle = 0;
    };
    std::mutex frame_event_mutex_;
    std::condition_variable frame_event_cv_;
    std::map<uint32_t, ScheduledFrameEvent> frame_events_;
    bool frame_event_stop_ = false;
    std::thread frame_event_thread_;
    void run_frame_events();
    void stop_frame_event_thread();
    std::atomic<uint64_t> xbox_counters_[static_cast<size_t>(xbox::Counter::Count)] = {};
    std::atomic<uint64_t> last_frame_token_{0};
    struct AliasWindow {
        uint64_t gpu_base, cpu_base, size;
    };
    mutable std::mutex alias_mutex_;
    std::vector<AliasWindow> aliases_;
    uint64_t alias_listener_ = 0;  // os::xmem_windows subscription (O27)
};

// ---------------------------------------------------------------------------------------------
// Children
// ---------------------------------------------------------------------------------------------

class Blob final : public com::ComObject, public IBlobHost {
public:
    static constexpr const char* kComClassName = "ID3DBlob";
    explicit Blob(uint64_t size);
    ~Blob() override;
    bool valid() const { return buffer_ != 0 || size_ == 0; }
    uint64_t GetBufferPointer() override { return buffer_; }
    uint64_t GetBufferSize() override { return size_; }
    // Creates a blob holding `bytes` (guest memory); empty Ref on allocation failure.
    static com::Ref<Blob> create(const void* bytes, size_t size);

private:
    uint64_t buffer_ = 0;
    uint64_t size_ = 0;
};

class CommandAllocator final : public ChildBase<ICommandAllocatorHost> {
public:
    static constexpr const char* kComClassName = "ID3D12CommandAllocator";
    CommandAllocator(com::Ref<Device> device, uint32_t type);
    ~CommandAllocator() override;
    HResult Reset() override;
    // Xbox slot 9: the executable registers a callback that hands out memory for the allocator's
    // command buffers. This layer never writes command buffers into guest memory, so the callback
    // is only recorded (never called).
    HResult SetAllocationCallback_unverified(uint64_t callback, uint64_t context) override;
    uint32_t type() const { return type_; }
    uint64_t resets() const { return resets_.load(); }
    uint64_t allocation_callback() const { return callback_.load(); }

private:
    uint32_t type_;
    std::atomic<uint64_t> resets_{0};
    std::atomic<uint64_t> callback_{0};
};

class Fence final : public ChildBase<IFenceHost> {
public:
    static constexpr const char* kComClassName = "ID3D12Fence";
    Fence(com::Ref<Device> device, uint64_t initial_value, uint32_t flags);
    ~Fence() override;
    uint64_t GetCompletedValue() override { return published_.load(std::memory_order_acquire); }
    HResult SetEventOnCompletion(uint64_t value, uint64_t event_handle) override;
    HResult Signal(uint64_t value) override;
    uint32_t GetCreationFlags() override { return flags_; }
    uint64_t CompletedValueAddress() override;
    // GPU-side signal (queue Signal): the work submitted before it has been handed off (and its
    // CPU-visible memory read) at once, so the value is executed now; it is published now or, with
    // DeviceOptions::fence_publish_lag, at Device::publish_fences (next frame presented / cap).
    void signal_from_queue(uint64_t value, uint64_t present_epoch);
    // The executed value (what has really been handed off), and the published one.
    uint64_t executed_value() const { return value_.load(std::memory_order_acquire); }
    // Publishes the pending queue signals the rule admits: signalled at an epoch at least
    // `min_lag_frames` below `present_epoch` (default 1: the frame's own PresentX), or older than
    // `cap`. Accumulates the counts published by present/cap and the total elapsed presents.
    void publish_due(uint64_t present_epoch, uint32_t min_lag_frames, std::chrono::steady_clock::time_point now,
                     std::chrono::milliseconds cap, uint64_t* by_present, uint64_t* by_cap, uint64_t* lag_frames);
    // Publishes `value` (if executed) because a blocking wait needs it; true when it was pending.
    bool publish_for_wait(uint64_t value);

private:
    void fire_events_locked(std::vector<uint64_t>* to_set);
    void publish_value_locked();
    void publish_locked(uint64_t value);

    struct PendingSignal {
        uint64_t value;
        uint64_t epoch;
        std::chrono::steady_clock::time_point at;
    };
    std::atomic<uint64_t> value_;      // executed
    std::atomic<uint64_t> published_;  // what the title sees (word, GetCompletedValue)
    std::deque<PendingSignal> pending_;
    bool registered_ = false;
    uint32_t flags_;
    uint64_t value_word_ = 0;  // guest word behind CompletedValueAddress (allocated on first use)
    bool destroyed_ = false;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::multimap<uint64_t, uint64_t> waits_;  // value -> event handle
};

class DescriptorHeap final : public ChildBase<IDescriptorHeapHost> {
public:
    static constexpr const char* kComClassName = "ID3D12DescriptorHeap";
    DescriptorHeap(com::Ref<Device> device, d3d::DescriptorHeap* heap, const GuestDescriptorHeapDesc& desc);
    ~DescriptorHeap() override;
    GuestDescriptorHeapDesc GetDesc() override { return desc_; }
    CpuHandleResult GetCPUDescriptorHandleForHeapStart() override { return {heap_->cpu_start().ptr}; }
    GpuHandleResult GetGPUDescriptorHandleForHeapStart() override { return {heap_->gpu_start().ptr}; }
    d3d::DescriptorHeap* heap() const { return heap_; }
    // Guest memory behind the CPU handles that the device allocated for this heap (freed with it).
    void set_owned_memory(uint64_t va) { owned_memory_ = va; }

private:
    d3d::DescriptorHeap* heap_;
    GuestDescriptorHeapDesc desc_;
    uint64_t owned_memory_ = 0;
};

class RootSignature final : public ChildBase<IRootSignatureHost> {
public:
    static constexpr const char* kComClassName = "ID3D12RootSignature";
    RootSignature(com::Ref<Device> device, std::unique_ptr<d3d::RootSignature> signature);
    ~RootSignature() override;
    const d3d::RootSignature& signature() const { return *signature_; }

private:
    std::unique_ptr<d3d::RootSignature> signature_;
};

class QueryHeap final : public ChildBase<IQueryHeapHost> {
public:
    static constexpr const char* kComClassName = "ID3D12QueryHeap";
    QueryHeap(com::Ref<Device> device, const GuestQueryHeapDesc& desc);
    ~QueryHeap() override;
    const GuestQueryHeapDesc& desc() const { return desc_; }
    // The consumer's query set (kGpuOpcodeCreateQuerySet); 0: none (no consumer support or an
    // unsupported heap type), the heap's queries are then counted and never written.
    uint32_t query_set_id() const { return query_set_id_; }
    uint32_t query_set_kind() const { return query_set_kind_; }

private:
    GuestQueryHeapDesc desc_;
    uint32_t query_set_id_ = 0;
    uint32_t query_set_kind_ = 0;
};

class CommandSignature final : public ChildBase<ICommandSignatureHost> {
public:
    static constexpr const char* kComClassName = "ID3D12CommandSignature";
    // D3D12_INDIRECT_ARGUMENT_TYPE of the work argument (the last one); kIndirectArgXbox105 is an
    // Xbox-only kind the title uses (meaning not established).
    static constexpr uint32_t kIndirectArgDraw = 0;
    static constexpr uint32_t kIndirectArgDrawIndexed = 1;
    static constexpr uint32_t kIndirectArgDispatch = 2;
    static constexpr uint32_t kIndirectArgDispatchMesh = 10;
    static constexpr uint32_t kIndirectArgXbox105 = 105;
    static constexpr uint32_t kIndirectArgUnknown = 0xFFFFFFFFu;
    // D3D12_INDIRECT_ARGUMENT_TYPE of the arguments that change state before the work argument.
    static constexpr uint32_t kIndirectArgVertexBufferView = 3;
    static constexpr uint32_t kIndirectArgIndexBufferView = 4;
    static constexpr uint32_t kIndirectArgConstant = 5;
    static constexpr uint32_t kIndirectArgCbv = 6;
    static constexpr uint32_t kIndirectArgSrv = 7;
    static constexpr uint32_t kIndirectArgUav = 8;
    // One argument of the signature: Constant carries (root_parameter, dest_word, num_words).
    struct Arg {
        uint32_t type = kIndirectArgUnknown;
        uint32_t root_parameter = 0;
        uint32_t dest_word = 0;
        uint32_t num_words = 0;
    };
    CommandSignature(com::Ref<Device> device, uint32_t byte_stride, uint32_t argument_count,
                     uint32_t work_type = kIndirectArgUnknown, std::vector<Arg> args = {})
        : ChildBase(std::move(device)), byte_stride_(byte_stride), argument_count_(argument_count),
          work_type_(work_type), args_(std::move(args)) {}
    uint32_t byte_stride() const { return byte_stride_; }
    uint32_t argument_count() const { return argument_count_; }
    uint32_t work_type() const { return work_type_; }
    // The arguments in record order (the last is the work argument); empty when unparsed.
    const std::vector<Arg>& args() const { return args_; }
    // Bytes one argument occupies in a record (arguments are tightly packed, in order).
    static uint32_t arg_bytes(const Arg& a) {
        switch (a.type) {
        case kIndirectArgDraw: return 16;
        case kIndirectArgDrawIndexed: return 20;
        case kIndirectArgDispatch:
        case kIndirectArgDispatchMesh:
        case kIndirectArgXbox105: return 12;
        case kIndirectArgVertexBufferView:
        case kIndirectArgIndexBufferView: return 16;
        case kIndirectArgConstant: return 4u * a.num_words;
        case kIndirectArgCbv:
        case kIndirectArgSrv:
        case kIndirectArgUav: return 8;
        default: return 0;
        }
    }

private:
    uint32_t byte_stride_;
    uint32_t argument_count_;
    uint32_t work_type_;
    std::vector<Arg> args_;
};

class CommandQueue final : public ChildBase<ICommandQueueHost> {
public:
    static constexpr const char* kComClassName = "ID3D12CommandQueue";
    CommandQueue(com::Ref<Device> device, const GuestCommandQueueDesc& desc);
    ~CommandQueue() override;
    HResult Signal(InterfaceArg<IFenceHost> fence, uint64_t value) override;
    HResult Wait(InterfaceArg<IFenceHost> fence, uint64_t value) override;
    HResult GetTimestampFrequency(GuestPtr<uint64_t> frequency) override;
    // Xbox slot 32 (d3d12_xs_xbox.cpp): plane 0's first resource is the back buffer.
    HResult PresentX(uint32_t plane_count, uint64_t planes, uint64_t parameters) override;
    // Xbox slot 35: (count, command lists, frame token, flags); the token is recorded, the flags are
    // not interpreted.
    void ExecuteCommandListsX(com::CallContext& ctx) override;
    HResult SuspendX(uint32_t flags) override;
    HResult ResumeX() override;
    HResult Slot31_UpdatePageMappings_unverified(com::CallContext& ctx) override;
    bool suspended() const { return suspended_.load(); }
    // ExecuteCommandLists semantics for host callers and tests: list interface pointers.
    void execute(const std::vector<uint64_t>& lists);
    const GuestCommandQueueDesc& desc() const { return desc_; }

private:
    GuestCommandQueueDesc desc_;
    std::atomic<bool> suspended_{false};
};

// ---------------------------------------------------------------------------------------------
// Interface declarations (vtables)
// ---------------------------------------------------------------------------------------------

extern const com::Interface<IDeviceHost> kID3D12Device;
extern const com::Interface<ICommandQueueHost> kID3D12CommandQueue;
extern const com::Interface<ICommandAllocatorHost> kID3D12CommandAllocator;
extern const com::Interface<IFenceHost> kID3D12Fence;
extern const com::Interface<IDescriptorHeapHost> kID3D12DescriptorHeap;
extern const com::Interface<IRootSignatureHost> kID3D12RootSignature;
extern const com::Interface<IQueryHeapHost> kID3D12QueryHeap;
extern const com::Interface<ICommandSignatureHost> kID3D12CommandSignature;
extern const com::Interface<IBlobHost> kID3DBlob;

// DXGI format block layout (bytes per element or compressed block, block size, plane count).
struct DxgiFormatLayout {
    uint32_t block_bytes = 0;
    uint32_t block_width = 1;
    uint32_t block_height = 1;
    uint32_t planes = 1;
    bool known = false;
};
DxgiFormatLayout dxgi_format_layout(uint32_t format);
// Bytes per element of `plane` (depth-stencil formats: 4-byte depth plane, 1-byte stencil plane).
uint32_t plane_block_bytes(uint32_t format, uint32_t plane);
// Size of a resource's subresources: the GFX10 placement for verified console layouts and for
// small-placement textures (see resource_allocation), otherwise the linear size (no tiling or
// alignment padding).
uint64_t resource_size_estimate(const GuestResourceDesc& desc);
// GetResourceAllocationInfo's answer for one description. A Texture2D with layout UNKNOWN that
// asks for the 4 KiB small-resource alignment is sized in 4 KiB swizzle blocks (SW_4KB_S) and
// keeps 4 KiB only while that surface fits in 64 KiB; otherwise the answer is the 64 KiB default
// (the title then repeats the query with alignment 0, as D3D12 applications do).
struct ResourceAllocation {
    uint64_t size = 0;
    uint64_t alignment = 0;
};
ResourceAllocation resource_allocation(const GuestResourceDesc& desc);

// Process-wide totals of the work the title records, by outcome (monotonic; sampled per Present).
// A draw is "issued" when the title records it and ends in exactly one of: emitted (a packet),
// no shader module, mesh/amplification pipeline (placeholder state), rectangle list, or dropped by
// the state bridge (no pipeline state, missing bindings, ...). Dispatches likewise. Indirect
// executions (ExecuteIndirect) are counted separately: emitted or skipped. Uploads count WriteTexture / WriteBuffer
// packets the layer builds from guest memory; copies it cannot express are "copies skipped".
enum class Work : size_t {
    DrawsIssued,
    DrawsEmitted,
    DrawsNoShaderModule,
    DrawsMeshPipeline,
    DrawsRectList,
    DrawsDroppedByBridge,
    DispatchesIssued,
    DispatchesEmitted,
    DispatchesNoShaderModule,
    DispatchesDroppedByBridge,
    IndirectExecutionsSkipped,
    IndirectExecutionsEmitted,  // G66: draws / indexed draws / dispatches through the clamp pass
    IndirectVertexIndexArgsSkipped,  // ... signatures changing vertex / index buffer views per command (also counted in ...Skipped)
    IndirectRootViewArgsSkipped,     // ... signatures setting root CBV / SRV / UAV addresses per command (also counted in ...Skipped)
    TextureUploads,
    TextureSyncViewSkips,    // sync skipped: view-backed range, no page written since the last full check
    TextureSyncViewTracked,  // sync of a wholly view-backed range (skipped or not)
    TextureSyncUntracked,    // sync of a range with a host pointer but no view coverage
    TextureSyncOwnedChecks,  // binds of a texture whose texels our read-back re-tile wrote
    TextureSyncOwnedHashSkips,  // ... where no page was written since the last check (no surface hash)
    BufferUploads,
    CopiesSkipped,
    // Texture-to-texture copies skipped, by reason (each is also counted in CopiesSkipped).
    CopySkipGateOff,       // CopyTexture packet disabled (gpu_texture_copy() off, e.g. a GPU capture)
    CopySkipNoGpuTexture,  // source or destination has no GPU texture object
    CopySkipMsaa,          // multisampled source or destination
    CopySkipSwapchain,     // swapchain / canvas texture
    CopySkipDimension,     // different or unsupported dimensions (1D, 2D vs 3D)
    CopySkipFormat,        // formats not copy-compatible
    CopySkipOverlap,       // overlapping same-subresource copy
    CopySkipEmitFailed,    // planned, but the emit step refused it
    ResourcesCreated,
    ResourcesDestroyed,
    // Reflected descriptor-table bindings joined by table slot because the root signature does not
    // declare that slot as their class (stage_reflection.h, #482), per pipeline state created.
    ReflectionTableSlotsOutsideRs,
    // Lazy GPU objects (G66, #1077; Device::defer_create): guest resources whose create packet was
    // held at creation, written before the first packet that used them, and dropped at release
    // without ever being written.
    GpuObjectsDeferred,
    GpuObjectsCreatedOnUse,
    GpuObjectsNeverCreated,
    // Lane-indexed heap bindings (VGPR / waterfall bindless, stage_reflection.h, G62 #482) that
    // failed their pipeline's reflection join because the bridge has no descriptor arrays yet, per
    // pipeline state created.
    ReflectionLaneIndexedUnbound,
    // G66: per-draw/dispatch object reuse. Bind group lookups served from the cache vs rebuilt, and
    // the GPU objects the recorded work created (a steady scene creates none after warm-up).
    BindGroupCacheHits,
    BindGroupCacheMisses,
    BindGroupsCreated,
    PipelinesCreated,
    // Reflected root-descriptor bindings whose user SGPR starts a descriptor table, joined as a
    // table slot (stage_reflection.h, G62 #482), per pipeline state created.
    ReflectionRootDescriptorTables,
    ReflectionVertexInputsDefaulted,
    // Reflected bindings whose descriptor origin the translator could not trace ("unresolved"
    // sources, sidecar version 3, G62 #482), bound as the worker's default, per pipeline state created.
    ReflectionUnresolvedBindings,
    // G66 GPU -> CPU read-back (d3d12_readback.h): ReadbackBuffer / ReadbackTexture packets issued
    // and the bytes they write into guest memory, completions the consumer reported, queue signals
    // held until their read-backs landed, and holds ended by the timeout instead.
    ReadbacksIssued,
    ReadbackBytes,
    ReadbacksCompleted,
    ReadbackSignalsHeld,
    ReadbackTimeouts,
    // G66 partly-held mip chains: sampled texture bindings emitted with their base mip level raised
    // to the first level the title holds (streamed in / mapped) or to the view's min-LOD clamp.
    MipClampedBinds,
    // Backlog 35: storage-texture bindings of a depth texture that got no colour twin (their writes
    // are lost), those bound to a twin, and the depth -> twin / twin -> depth copies emitted.
    StorageViewsDepthUntwinned,
    DepthTwinBindings,
    DepthTwinCopiesIn,
    DepthTwinCopiesOut,
    // Backlog 8: compute-written float textures read back and re-tiled into guest memory: read-backs
    // recorded, re-tiled into guest memory (real GPU bytes only), and discarded because the consumer
    // completed them without writing (or the guest surface could not be addressed).
    ComputeTextureReadbacks,
    ComputeTextureRetiled,
    ComputeTextureRetileDiscarded,
    // Backlog 35, stencil plane: bindings bound to an unsigned-integer twin, copies in / out, masked draws.
    StencilTwinBindings,
    StencilTwinCopiesIn,
    StencilTwinCopiesOut,
    StencilTwinDraws,
    // CPU-visible buffer flush at ExecuteCommandLists (flush_shadow): bytes compared with the shadow,
    // and the whole-buffer comparisons (no write tracking for the buffer) with their bytes.
    ShadowComparedBytes,
    ShadowFullCompares,
    ShadowFullCompareBytes,
    // #1963 page-indexed native flush (flush_cpu_buffers): host ranges taken at each flush (and the
    // takes that could not tell, whose entries were all flushed), pages reported written, buffers
    // flushed / skipped / flushed for want of write tracking, uniform ranges compared / skipped, and
    // the --flush-crosscheck results: an entry skipped although it changed that the next flush's take
    // reported (a write racing with the flush, flushed one submission later as before) or did not (a miss).
    FlushIndexTakes,
    FlushIndexTakeFailures,
    FlushIndexDirtyPages,
    FlushBuffersVisited,
    FlushBuffersSkipped,
    FlushBuffersUntracked,
    FlushUniformsCompared,
    FlushUniformsSkipped,
    FlushStorageCompared,  // relocated storage ranges looked at / skipped by an indexed flush
    FlushStorageSkipped,
    // relocate_storage_buffer misses that placed a range, and the present groups / capacity buckets / LRU
    // nodes their indexed reuse lookups looked at (gpu/storage_reuse_lru.h).
    StorageRelocationMisses,
    StorageReuseScanSteps,
    FlushIaCompared,       // relocated IA (vertex / index) ranges looked at / skipped by an indexed flush
    FlushIaSkipped,
    FlushCrosscheckLate,
    FlushCrosscheckMisses,
    // Indexed flushes that were not cross-checked: calls and wall time (ns) in flush_cpu_buffers.
    FlushIndexedCalls,
    FlushIndexedNs,
    FlushBufferNs,   // indexed flush time spent on buffers (take included) / on uniform ranges
    FlushUniformNs,
    // Query heaps (BeginQuery / EndQuery / ResolveQueryData): calls are counted by D3D12_QUERY_TYPE group.
    // Occlusion and timestamp heaps run on the consumer's query sets (gpu_queries()); every other result
    // is left unwritten (never faked), so a title reading it sees whatever the buffer held before.
    QueryBegins,
    QueryEnds,
    QueryResolves,
    QueryOcclusionOps,  // OCCLUSION (0) and BINARY_OCCLUSION (1)
    QueryTimestampOps,  // TIMESTAMP (2)
    QueryOtherOps,      // pipeline statistics, stream output, anything else
    QueryResultsUnwritten,  // query results a resolve was asked for and could not write
    QueryPackets,           // begin / end / resolve packets for the consumer's query sets
    QueryResultsWritten,    // query results resolved into a buffer by the consumer
    Count
};
// #1963: how flush_cpu_buffers picks what to compare on native builds. Index (default): only the
// buffers / uniform ranges whose host pages the OS write tracking reports written. Full: every one
// (the earlier behaviour; --flush-full). CrossCheck: every one, and a skip the index would have made
// for an entry that changed is counted (--flush-crosscheck; Work::FlushCrosscheck*).
enum class FlushMode : uint8_t { Index, Full, CrossCheck };
void set_flush_mode(FlushMode mode);
FlushMode flush_mode();
// CrossCheck: only every n-th flush compares everything (default 1: all); the others run as Index.
void set_flush_crosscheck_every(uint32_t n);
void count_work(Work w, uint64_t n = 1);
uint64_t work_total(Work w);
const char* work_name(Work w);
// Bytes the console's D3D12 allocates for a committed resource: the 64 KiB-block surface size
// for the 2D surfaces os/surface_blocks.h covers, otherwise resource_size_estimate.
uint64_t committed_memory_size(const GuestResourceDesc& desc);

// XMemAlloc attributes for committed-resource memory (XALLOC_ATTRIBUTES fields):
//   allocator id (bits 16..23) = 0x80 - inferred from a title's allocator hook, which registers
//     its D3D resource arena handler for allocator id 0x80 (the public layout reserves ids from
//     128 for the system; the first is the D3D allocator);
//   page size (bits 14..15) = 2 - inferred from the same hook, which accepts only page-size field
//     2 (large pages) for that arena;
//   object type (bits 0..13), alignment, memory type and the rest = 0 (not documented values).
inline constexpr uint64_t kD3D12ResourceXMemAttributes = (uint64_t{0x80} << 16) | (uint64_t{2} << 14);
com::Ref<Device> device_ref(Device* device);

class CommandList;

// G79: Read-only snapshot of a command list's validated 3D dynamic state and depth-twin sync counts.
struct CommandListDynamicStateSnapshot {
    std::vector<pso::D3D12Viewport> viewports;
    std::vector<pso::D3D12Rect> scissors;
    float blend_factor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    uint32_t stencil_ref = 0;
    uint32_t bound_dsv_texture = 0;
    uint32_t bound_rt_width = 0;
    uint32_t bound_rt_height = 0;
    uint64_t depth_srv_sync_flushes = 0;
    uint64_t depth_srv_sync_refreshes = 0;
    uint64_t resolves_emitted = 0;
    uint64_t resolves_fallback = 0;
};
CommandListDynamicStateSnapshot inspect_command_list_state(const CommandList* list);
const bridge::DepthTwinTracker* inspect_command_list_depth_twin(const CommandList* list);

// Root signature description from a D3D12_VERSIONED_ROOT_SIGNATURE_DESC in guest memory (all nested
// arrays read through the seam). False + message on malformed input.
bool read_versioned_root_signature(uint64_t desc_va, d3d::RootSignatureDesc* out, std::string* error);
bool read_versioned_root_signature(const GuestVersionedRootSignatureDesc& desc, d3d::RootSignatureDesc* out,
                                   std::string* error);

// Hands `object` to guest code as the interface answering `riid` (QueryInterface semantics): writes
// the pointer to *out and returns S_OK, or E_NOINTERFACE (and *out = 0). Consumes the reference.
template <typename T>
HResult hand_out(com::Ref<T> object, const Guid& riid, OutPtr out) {
    if (!object) {
        if (out) {
            out.write(0);
        }
        return kEOutOfMemory;
    }
    if (out.is_null()) {
        return kSFalse;  // D3D12: a null out pointer only tests whether creation would succeed
    }
    uint64_t pointer = 0;
    const HResult hr = object->query_interface(riid, &pointer);
    if (hr < 0 || !out.write(pointer)) {
        if (pointer != 0) {
            object->release();
        }
        out.write(0);
        return hr < 0 ? hr : kEPointer;
    }
    return kSOk;
}

template <typename Iface>
HResult ChildBase<Iface>::GetDevice(const Guid& riid, OutPtr out) {
    if (out.is_null()) {
        return kEPointer;
    }
    uint64_t pointer = 0;
    const HResult hr = device_->query_interface(riid, &pointer);
    out.write(pointer);
    return hr;
}

}  // namespace playgta6::gpu::d3d12_guest
