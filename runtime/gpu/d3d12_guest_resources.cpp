#include "gpu/d3d12_guest_resources.h"
#include "gpu/gpu_va_watch.h"
#include "gpu/pipeline_prewarm.h"
#include "gpu/prewarm_pipeline_desc.h"
#include "gpu/small_texture_pool.h"
#include "gpu/texture_budget.h"
#include "gpu/xbox_embedded_native.h"

#include <optional>
#include <algorithm>
#if defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "gpu/gfx10_detile.h"
#include "os/guest_window.h"
#include "os/thread_spans.h"
#include "os/os_shim.h"
#include "os/win32_kernel.h"
#include "os/vmem.h"
#include "os/surface_blocks.h"
#include "os/xbox_scarlett.h"

namespace playgta6::gpu::d3d12_guest {

namespace {

// A texture known only by a view at an address (Device::ensure_address_texture): the first 64 KiB
// of its memory, read through the CPU window.
struct ViewProbe {
    uint64_t cpu = 0;
    uint64_t len = 0;
    bool seen_data = false;
};

struct TextureProbe {
    std::mutex mutex;
    std::vector<Resource*> textures;
    size_t cursor = 0;
    uint64_t with_data = 0;
    std::vector<ViewProbe> views;
    size_t view_cursor = 0;
    uint64_t views_with_data = 0;
};

TextureProbe& texture_probe() {
    static TextureProbe p;
    return p;
}

constexpr uint64_t kShadowPage = 4096;


constexpr uint64_t kMaxWriteChunk = 1u << 20;
constexpr uint64_t kMaxShaderBytes = 64u << 20;
constexpr uint32_t kMaxInputElements = 32;

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%" PRIx64, v);
    return b;
}

constexpr uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

// Block geometry of a WebGPU host texture format (used when zero-filling reused pooled host
// textures whose DXGI format was promoted to a wider WebGPU format, e.g. A8_UNORM -> rgba8unorm
// or R16_UNORM -> r32float, so WriteTexture bytes_per_row matches the host format).
DxgiFormatLayout wgpu_format_block_layout(pso::WgpuTextureFormat format) noexcept {
    using F = pso::WgpuTextureFormat;
    auto uncomp = [](uint32_t bpp) noexcept { return DxgiFormatLayout{bpp, 1u, 1u, 1u, true}; };
    auto bc = [](uint32_t bpb) noexcept { return DxgiFormatLayout{bpb, 4u, 4u, 1u, true}; };
    switch (format) {
        case F::R8Unorm:
        case F::R8Snorm:
        case F::R8Uint:
        case F::R8Sint:
        case F::Stencil8:
            return uncomp(1u);
        case F::R16Unorm:
        case F::R16Snorm:
        case F::R16Uint:
        case F::R16Sint:
        case F::R16Float:
        case F::Rg8Unorm:
        case F::Rg8Snorm:
        case F::Rg8Uint:
        case F::Rg8Sint:
        case F::Depth16Unorm:
            return uncomp(2u);
        case F::R32Uint:
        case F::R32Sint:
        case F::R32Float:
        case F::Rg16Unorm:
        case F::Rg16Snorm:
        case F::Rg16Uint:
        case F::Rg16Sint:
        case F::Rg16Float:
        case F::Rgba8Unorm:
        case F::Rgba8UnormSrgb:
        case F::Rgba8Snorm:
        case F::Rgba8Uint:
        case F::Rgba8Sint:
        case F::Bgra8Unorm:
        case F::Bgra8UnormSrgb:
        case F::Rgb9e5Ufloat:
        case F::Rgb10a2Uint:
        case F::Rgb10a2Unorm:
        case F::Rg11b10Ufloat:
        case F::Depth24Plus:
        case F::Depth24PlusStencil8:
        case F::Depth32Float:
            return uncomp(4u);
        case F::Rg32Uint:
        case F::Rg32Sint:
        case F::Rg32Float:
        case F::Rgba16Unorm:
        case F::Rgba16Snorm:
        case F::Rgba16Uint:
        case F::Rgba16Sint:
        case F::Rgba16Float:
        case F::Depth32FloatStencil8:
            return uncomp(8u);
        case F::Rgba32Uint:
        case F::Rgba32Sint:
        case F::Rgba32Float:
            return uncomp(16u);
        case F::Bc1RgbaUnorm:
        case F::Bc1RgbaUnormSrgb:
        case F::Bc4RUnorm:
        case F::Bc4RSnorm:
            return bc(8u);
        case F::Bc2RgbaUnorm:
        case F::Bc2RgbaUnormSrgb:
        case F::Bc3RgbaUnorm:
        case F::Bc3RgbaUnormSrgb:
        case F::Bc5RgUnorm:
        case F::Bc5RgSnorm:
        case F::Bc6hRgbUfloat:
        case F::Bc6hRgbFloat:
        case F::Bc7RgbaUnorm:
        case F::Bc7RgbaUnormSrgb:
            return bc(16u);
        default:
            return DxgiFormatLayout{};
    }
}

bool heap_cpu_visible(const GuestHeapProperties& p) {
    if (p.Type == HeapType::Upload || p.Type == HeapType::Readback) {
        return true;
    }
    // CUSTOM: CPUPageProperty 2 WRITE_COMBINE / 3 WRITE_BACK are CPU accessible.
    return p.Type == HeapType::Custom && (p.CPUPageProperty == 2 || p.CPUPageProperty == 3);
}

uint32_t full_mip_count(const GuestResourceDesc& d) {
    uint64_t m = std::max<uint64_t>({d.Width, d.Height,
                                     d.Dimension == ResourceDimension::Texture3D ? d.DepthOrArraySize : 1u});
    uint32_t n = 1;
    while (m > 1) {
        m >>= 1;
        ++n;
    }
    return n;
}

float f32(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint64_t u64_at(const uint32_t* u, size_t i) {
    return uint64_t{u[i]} | (uint64_t{u[i + 1]} << 32);
}

// Formats WebGPU allows STORAGE_BINDING for: the core set, plus the "texture-formats-tier1" set
// and BGRA8 with "bgra8unorm-storage". Both texture creation paths (committed/placed resources and
// address-only textures) gate STORAGE_BINDING through this one predicate.
bool storage_capable(pso::WgpuTextureFormat f, uint32_t features) {
    using F = pso::WgpuTextureFormat;
    switch (f) {
        case F::Rgba8Unorm: case F::Rgba8Snorm: case F::Rgba8Uint: case F::Rgba8Sint:
        case F::Rgba16Uint: case F::Rgba16Sint: case F::Rgba16Float:
        case F::R32Uint: case F::R32Sint: case F::R32Float:
        case F::Rg32Uint: case F::Rg32Sint: case F::Rg32Float:
        case F::Rgba32Uint: case F::Rgba32Sint: case F::Rgba32Float:
            return true;
        case F::R8Unorm: case F::R8Snorm: case F::R8Uint: case F::R8Sint:
        case F::Rg8Unorm: case F::Rg8Snorm: case F::Rg8Uint: case F::Rg8Sint:
        case F::R16Unorm: case F::R16Snorm: case F::R16Uint: case F::R16Sint: case F::R16Float:
        case F::Rg16Unorm: case F::Rg16Snorm: case F::Rg16Uint: case F::Rg16Sint: case F::Rg16Float:
        case F::Rgba16Unorm: case F::Rgba16Snorm:
        case F::Rgb10a2Uint: case F::Rgb10a2Unorm: case F::Rg11b10Ufloat:
            return (features & pso::WgpuFeature::TextureFormatsTier1) != 0;
        case F::Bgra8Unorm:
            return (features & pso::WgpuFeature::Bgra8UnormStorage) != 0;
        default:
            return false;
    }
}

// WebGPU forbids STORAGE_BINDING on *-srgb base textures. When a single-sample sRGB texture
// requests ALLOW_UNORDERED_ACCESS and its linear sibling supports STORAGE_BINDING under `features`,
// lower the host texture's base format to the linear sibling (listing the *-srgb sibling in
// view_formats so RTV/SRV/Present paths view it as sRGB while UAVs bind the linear base natively).
pso::WgpuTextureFormat storage_base_format(pso::WgpuTextureFormat mapped, bool allow_uav, uint32_t sample_count,
                                           uint32_t features) {
    if (!allow_uav || sample_count > 1 || storage_capable(mapped, features) || !bridge::is_srgb_format(mapped)) {
        return mapped;
    }
    const pso::WgpuTextureFormat linear = pso::srgb_sibling(mapped);
    if (linear != pso::WgpuTextureFormat::Undefined && storage_capable(linear, features)) {
        return linear;
    }
    return mapped;
}

// Renderable float/color formats that lack STORAGE_BINDING when optional WebGPU features
// (texture-formats-tier1 / bgra8unorm-storage) are absent: giving a 2D UAV texture RenderAttachment
// usage lets state_bridge copy UAV results back via a non-quantizing fullscreen render pass instead
// of dropping compute dispatches.
bool renderable_uav_fallback(pso::WgpuTextureFormat f, uint32_t features) {
    using F = pso::WgpuTextureFormat;
    switch (f) {
        case F::Bgra8Unorm:
        case F::Bgra8UnormSrgb:
        case F::R8Unorm:
        case F::Rg8Unorm:
        case F::R16Float:
        case F::Rg16Float:
        case F::Rgb10a2Unorm:
            return true;
        case F::Rg11b10Ufloat:
            return (features & pso::WgpuFeature::Rg11b10UfloatRenderable) != 0;
        default:
            return false;
    }
}

// CPU upload wideners for DXGI formats promoted to wider WebGPU formats:
//   A8_UNORM (1B)                    -> Rgba8Unorm (4B)
//   B5G6R5_UNORM (2B)                -> Rgba8Unorm (4B)
//   B5G5R5A1_UNORM (2B)              -> Rgba8Unorm (4B)
//   B4G4R4A4_UNORM (2B)              -> Rgba8Unorm (4B)
//   R10G10B10_7E3_A2_FLOAT (116, 4B) -> Rgba16Float (8B)
//   R10G10B10_6E4_A2_FLOAT (117, 4B) -> Rgba16Float (8B)
struct CpuWidenSpec {
    uint32_t src_bpp = 0;
    uint32_t dst_bpp = 0;
};

CpuWidenSpec cpu_widen_spec(pso::DxgiFormat from, pso::WgpuTextureFormat to) noexcept {
    using D = pso::DxgiFormat;
    using F = pso::WgpuTextureFormat;
    if (to == F::Rgba8Unorm) {
        switch (from) {
            case D::A8_UNORM:
                return {1u, 4u};
            case D::B5G6R5_UNORM:
            case D::B5G5R5A1_UNORM:
            case D::B4G4R4A4_UNORM:
                return {2u, 4u};
            default:
                return {};
        }
    }
    if (to == F::Rgba16Float) {
        switch (from) {
            case D::R10G10B10_7E3_A2_FLOAT:
            case D::R10G10B10_6E4_A2_FLOAT:
                return {4u, 8u};
            default:
                return {};
        }
    }
    return {};
}

// Lossless 10-bit Xbox unsigned float 7e3 (bias 3, 7-bit mantissa, 3-bit exponent, finite [0, 31.875])
// -> IEEE 754 binary16. Every non-zero 7e3 value is an exact normal binary16 (min 2^-9 > 2^-14).
uint16_t decode_7e3_to_f16(uint32_t v) noexcept {
    const uint32_t e = (v >> 7) & 0x7u;
    const uint32_t m = v & 0x7Fu;
    if (e > 0u) {
        return static_cast<uint16_t>(((e + 12u) << 10) | (m << 3));
    }
    if (m == 0u) {
        return 0x0000u;
    }
    uint32_t msb = 0u;
    while ((m >> (msb + 1u)) != 0u) {
        ++msb;
    }
    const uint32_t exp = msb + 6u;  // (msb - 9) + 15
    const uint32_t mant = (m & ((1u << msb) - 1u)) << (10u - msb);
    return static_cast<uint16_t>((exp << 10) | mant);
}

// Lossless 10-bit Xbox unsigned float 6e4 (bias 7, 6-bit mantissa, 4-bit exponent, finite [0, 508.0])
// -> IEEE 754 binary16. Every non-zero 6e4 value is an exact normal binary16 (min 2^-12 > 2^-14).
uint16_t decode_6e4_to_f16(uint32_t v) noexcept {
    const uint32_t e = (v >> 6) & 0xFu;
    const uint32_t m = v & 0x3Fu;
    if (e > 0u) {
        return static_cast<uint16_t>(((e + 8u) << 10) | (m << 4));
    }
    if (m == 0u) {
        return 0x0000u;
    }
    uint32_t msb = 0u;
    while ((m >> (msb + 1u)) != 0u) {
        ++msb;
    }
    const uint32_t exp = msb + 3u;  // (msb - 12) + 15
    const uint32_t mant = (m & ((1u << msb) - 1u)) << (10u - msb);
    return static_cast<uint16_t>((exp << 10) | mant);
}

bool widen_cpu_texels(pso::DxgiFormat from, pso::WgpuTextureFormat to, const uint8_t* src, size_t count,
                      uint8_t* dst) noexcept {
    using D = pso::DxgiFormat;
    using F = pso::WgpuTextureFormat;
    if (src == nullptr || dst == nullptr) {
        return false;
    }
    if (from == D::A8_UNORM && to == F::Rgba8Unorm) {
        return pso::widen_a8_texels(from, to, src, count, dst);
    }
    if (to == F::Rgba8Unorm && from == D::B5G6R5_UNORM) {
        for (size_t i = 0; i < count; ++i) {
            uint16_t px = 0;
            std::memcpy(&px, src + i * 2, 2);
            const uint8_t b5 = static_cast<uint8_t>(px & 0x1Fu);
            const uint8_t g6 = static_cast<uint8_t>((px >> 5) & 0x3Fu);
            const uint8_t r5 = static_cast<uint8_t>((px >> 11) & 0x1Fu);
            dst[i * 4 + 0] = static_cast<uint8_t>((r5 << 3) | (r5 >> 2));
            dst[i * 4 + 1] = static_cast<uint8_t>((g6 << 2) | (g6 >> 4));
            dst[i * 4 + 2] = static_cast<uint8_t>((b5 << 3) | (b5 >> 2));
            dst[i * 4 + 3] = 0xFFu;
        }
        return true;
    }
    if (to == F::Rgba8Unorm && from == D::B5G5R5A1_UNORM) {
        for (size_t i = 0; i < count; ++i) {
            uint16_t px = 0;
            std::memcpy(&px, src + i * 2, 2);
            const uint8_t b5 = static_cast<uint8_t>(px & 0x1Fu);
            const uint8_t g5 = static_cast<uint8_t>((px >> 5) & 0x1Fu);
            const uint8_t r5 = static_cast<uint8_t>((px >> 10) & 0x1Fu);
            const uint8_t a1 = static_cast<uint8_t>((px >> 15) & 0x1u);
            dst[i * 4 + 0] = static_cast<uint8_t>((r5 << 3) | (r5 >> 2));
            dst[i * 4 + 1] = static_cast<uint8_t>((g5 << 3) | (g5 >> 2));
            dst[i * 4 + 2] = static_cast<uint8_t>((b5 << 3) | (b5 >> 2));
            dst[i * 4 + 3] = a1 != 0 ? 0xFFu : 0x00u;
        }
        return true;
    }
    if (to == F::Rgba8Unorm && from == D::B4G4R4A4_UNORM) {
        for (size_t i = 0; i < count; ++i) {
            uint16_t px = 0;
            std::memcpy(&px, src + i * 2, 2);
            const uint8_t b4 = static_cast<uint8_t>(px & 0xFu);
            const uint8_t g4 = static_cast<uint8_t>((px >> 4) & 0xFu);
            const uint8_t r4 = static_cast<uint8_t>((px >> 8) & 0xFu);
            const uint8_t a4 = static_cast<uint8_t>((px >> 12) & 0xFu);
            dst[i * 4 + 0] = static_cast<uint8_t>((r4 << 4) | r4);
            dst[i * 4 + 1] = static_cast<uint8_t>((g4 << 4) | g4);
            dst[i * 4 + 2] = static_cast<uint8_t>((b4 << 4) | b4);
            dst[i * 4 + 3] = static_cast<uint8_t>((a4 << 4) | a4);
        }
        return true;
    }
    if (to == F::Rgba16Float && (from == D::R10G10B10_7E3_A2_FLOAT || from == D::R10G10B10_6E4_A2_FLOAT)) {
        // Exact nearest-even binary16 values for 2-bit UNORM alpha {0, 1/3, 2/3, 1}.
        static constexpr uint16_t kA2ToHalf[4] = {0x0000u, 0x3555u, 0x3955u, 0x3C00u};
        const bool is_7e3 = from == D::R10G10B10_7E3_A2_FLOAT;
        for (size_t i = 0; i < count; ++i) {
            uint32_t px = 0;
            std::memcpy(&px, src + i * 4, 4);
            const uint32_t r10 = px & 0x3FFu;
            const uint32_t g10 = (px >> 10) & 0x3FFu;
            const uint32_t b10 = (px >> 20) & 0x3FFu;
            const uint32_t a2 = (px >> 30) & 0x3u;
            const uint16_t out[4] = {
                is_7e3 ? decode_7e3_to_f16(r10) : decode_6e4_to_f16(r10),
                is_7e3 ? decode_7e3_to_f16(g10) : decode_6e4_to_f16(g10),
                is_7e3 ? decode_7e3_to_f16(b10) : decode_6e4_to_f16(b10),
                kA2ToHalf[a2],
            };
            std::memcpy(dst + i * 8, out, 8);
        }
        return true;
    }
    return false;
}

bool widen_promoted_rows(pso::DxgiFormat from, pso::WgpuTextureFormat to, const uint8_t* src, size_t src_size,
                         uint32_t width, uint32_t rows, uint32_t slices, uint32_t src_pitch, uint64_t slice_pitch,
                         std::vector<uint8_t>* out, uint32_t* out_pitch) {
    const CpuWidenSpec spec = cpu_widen_spec(from, to);
    if (spec.src_bpp == 0 || rows == 0 || slices == 0) {
        return false;
    }
    const uint64_t src_row_bytes = uint64_t{width} * spec.src_bpp;
    const uint64_t pitch = align_up(uint64_t{width} * spec.dst_bpp, 256u);
    if (uint64_t{slices - 1} * slice_pitch + uint64_t{rows - 1} * src_pitch + src_row_bytes > src_size) {
        return false;
    }
    out->assign(static_cast<size_t>(pitch * rows * slices), 0);
    for (uint32_t z = 0; z < slices; ++z) {
        for (uint32_t r = 0; r < rows; ++r) {
            if (!widen_cpu_texels(from, to, src + z * slice_pitch + uint64_t{r} * src_pitch, width,
                                  out->data() + (uint64_t{z} * rows + r) * pitch)) {
                return false;
            }
        }
    }
    *out_pitch = static_cast<uint32_t>(pitch);
    return true;
}

uint32_t mips_or_all(uint32_t v) {
    return v == 0xFFFFFFFFu ? 0xffffu : v;
}

// Bytes of the whole console-layout surface from its base address (all mips and slices: on GFX10
// mip 0 is stored last, or inside the mip-tail block; see gfx10::MipPlacement). Returns 0 for
// D3D12 layouts 0-3 or unrecognized formats/modes.
uint64_t tiled_surface_bytes(uint32_t d3d12_layout, uint32_t dxgi_format, uint32_t width, uint32_t height,
                             uint32_t depth, uint32_t mip_levels, uint32_t array_size) noexcept {
    const gfx10::SwizzleMode mode = gfx10::decode_d3d12_layout(d3d12_layout);
    if (mode == gfx10::SwizzleMode::Unsupported ||
        (mode == gfx10::SwizzleMode::Linear && d3d12_layout != gfx10::kConsoleLinearLayout)) {
        return 0;
    }
    gfx10::TextureLayoutDesc desc{};
    desc.dxgi_format = dxgi_format;
    desc.width = width;
    desc.height = height;
    desc.depth = depth;
    desc.array_size = array_size;
    desc.mip_levels = std::max<uint32_t>(1u, mip_levels);
    desc.d3d12_layout = d3d12_layout;
    return gfx10::surface_bytes(desc);
}

bool read_guest_best_effort(uint64_t cpu_va, uint8_t* dst, size_t size) {
    if (size == 0) {
        return false;
    }
    if (com::read_guest(cpu_va, dst, size)) {
        return true;
    }
    uint8_t probe = 0;
    if (!com::read_guest(cpu_va, &probe, 1)) {
        return false;
    }
    std::memset(dst, 0, size);
    size_t off = 0;
    while (off < size) {
        const uint64_t cur = cpu_va + off;
        const size_t to_page_end = static_cast<size_t>(kShadowPage - (cur & (kShadowPage - 1)));
        const size_t step = std::min(size - off, to_page_end);
        if (!com::read_guest(cur, dst + off, step)) {
            if (off == 0) {
                dst[0] = probe;
            }
            break;
        }
        off += step;
    }
    return true;
}

}  // namespace

// O36: whether two byte ranges are equal. Without write tracking (wasm linear memory has none),
// flush_shadow compares every tracked CPU-visible buffer with its shadow at each submission: the
// real program tracks ~183 MB, compared twice a frame, the browser boot's frame-time bottleneck.
// The wasm64 libc memcmp compares 4 bytes per step; this compares 64 with wasm SIMD (the browser
// target requires SIMD anyway: lifted code is built with it), about 1.5x faster on the real
// program. The host memcmp is vectorized.
#if defined(__EMSCRIPTEN__)
__attribute__((target("simd128"))) bool same_bytes(const uint8_t* a, const uint8_t* b, size_t n) {
    typedef uint64_t u64x2 __attribute__((vector_size(16)));
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        u64x2 x[4];
        u64x2 y[4];
        std::memcpy(x, a + i, 64);
        std::memcpy(y, b + i, 64);
        const u64x2 d = (x[0] ^ y[0]) | (x[1] ^ y[1]) | (x[2] ^ y[2]) | (x[3] ^ y[3]);
        if ((d[0] | d[1]) != 0) {
            return false;
        }
    }
    for (; i < n; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}
#else
bool same_bytes(const uint8_t* a, const uint8_t* b, size_t n) {
    return std::memcmp(a, b, n) == 0;
}
#endif

namespace {

// R40: helper threads for the shadow comparison. Buffers without write tracking are compared whole
// at every submission: a memory-bandwidth-bound memcmp on the submitting (guest render) thread
// (before shared page views were tracked, os::take_written_pages, ~72% of its time in the menu
// flow: XMem physical pages are file views, which cannot carry MEM_WRITE_WATCH). Split
// across the caller and three helpers it runs ~1.7x faster in the boot (3.7 -> 6.1 GB/s; ~2x on
// an idle machine). run() hands out parts to the helpers and the caller alike and returns when all
// are done; concurrent callers take turns.
#if !defined(__EMSCRIPTEN__)
class ComparePool {
public:
    static ComparePool& get() {
        static ComparePool* pool = new ComparePool();  // never destroyed: helpers run until exit
        return *pool;
    }

    unsigned helpers() const { return static_cast<unsigned>(threads_.size()); }

    void run(unsigned parts, const std::function<void(unsigned)>& fn) {
        std::lock_guard<std::mutex> turn(run_mutex_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            fn_ = &fn;
            parts_ = parts;
            next_.store(0, std::memory_order_relaxed);
            done_.store(0, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
        }
        wake_.notify_all();
        drain(fn, parts);
        // What is left is at most one part per helper (tens of microseconds): spin rather than pay
        // a blocking wait's wake-up latency once per buffer.
        auto finished = [&] {
            return done_.load(std::memory_order_acquire) == parts && active_.load(std::memory_order_acquire) == 0;
        };
        spin_until(finished);
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, finished);  // (under the lock: no helper can join after this)
        fn_ = nullptr;                   // a helper waking late finds no work
    }

private:
    ComparePool() {
        const unsigned cores = std::thread::hardware_concurrency();
        const unsigned helpers = std::min(3u, cores / 4);  // bandwidth-bound: 4 threads get most of it
        for (unsigned i = 0; i < helpers; ++i) {
            threads_.emplace_back([this] { helper(); });
        }
        for (std::thread& t : threads_) {
            t.detach();
        }
    }

    void drain(const std::function<void(unsigned)>& fn, unsigned parts) {
        for (;;) {
            const unsigned k = next_.fetch_add(1, std::memory_order_relaxed);
            if (k >= parts) {
                return;
            }
            fn(k);
            if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 == parts) {
                std::lock_guard<std::mutex> lock(mutex_);
                finished_.notify_all();
            }
        }
    }

    // Spins (yielding) for up to ~200 us until `ready()`.
    template <typename Ready>
    static void spin_until(const Ready& ready) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
        while (!ready()) {
            if (std::chrono::steady_clock::now() > deadline) {
                return;
            }
            std::this_thread::yield();
        }
    }

    void helper() {
        uint64_t seen = 0;
        for (;;) {
            // A flush compares its buffers back to back: wait for the next one spinning first.
            spin_until([&] { return generation_.load(std::memory_order_acquire) != seen; });
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return generation_.load(std::memory_order_relaxed) != seen; });
            seen = generation_.load(std::memory_order_relaxed);
            if (fn_ == nullptr) {
                continue;
            }
            const std::function<void(unsigned)>* fn = fn_;
            const unsigned parts = parts_;
            active_.fetch_add(1, std::memory_order_relaxed);  // (under mutex_, while fn_ is set)
            lock.unlock();
            drain(*fn, parts);
            if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> relock(mutex_);
                finished_.notify_all();
            }
        }
    }

    std::mutex run_mutex_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable finished_;
    const std::function<void(unsigned)>* fn_ = nullptr;  // guarded by mutex_
    unsigned parts_ = 0;                                 // guarded by mutex_
    std::atomic<uint64_t> generation_{0};                // changed under mutex_
    std::atomic<unsigned> active_{0};                    // helpers inside drain(); joins under mutex_
    std::atomic<unsigned> next_{0};
    std::atomic<unsigned> done_{0};
    std::vector<std::thread> threads_;
};
#endif

}  // namespace

void compare_pages(size_t pages,
                   uint64_t bytes,
                   const std::function<bool(size_t)>& same,
                   std::vector<uint8_t>* changed) {
    changed->assign(pages, 0);
    auto compare = [&](size_t first, size_t last) {
        for (size_t i = first; i < last; ++i) {
            (*changed)[i] = same(i) ? 0 : 1;
        }
    };
#if !defined(__EMSCRIPTEN__)
    constexpr uint64_t kPartBytes = 256u << 10;
    constexpr uint64_t kParallelMinBytes = 1u << 20;
    ComparePool& pool = ComparePool::get();
    if (pool.helpers() != 0 && pages != 0 && bytes >= kParallelMinBytes) {
        const uint64_t page_size = std::max<uint64_t>(1, bytes / pages);
        const size_t per_part = static_cast<size_t>(std::max<uint64_t>(1, kPartBytes / page_size));
        const unsigned parts = static_cast<unsigned>((pages + per_part - 1) / per_part);
        pool.run(parts, [&](unsigned k) {
            const size_t first = static_cast<size_t>(k) * per_part;
            compare(first, std::min(pages, first + per_part));
        });
        return;
    }
#endif
    compare(0, pages);
}

uint64_t shader_identity(const void* data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ull;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 0x00000100000001b3ull;
    }
    return h;
}

namespace {

inline uint64_t load_u64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint64_t rotl64(uint64_t v, int r) {
    return (v << r) | (v >> (64 - r));
}

constexpr uint64_t kContentPrime1 = 0x9E3779B185EBCA87ull;
constexpr uint64_t kContentPrime2 = 0xC2B2AE3D27D4EB4Full;

inline uint64_t content_round(uint64_t acc, uint64_t word) {
    return rotl64(acc + word * kContentPrime2, 31) * kContentPrime1;
}

}  // namespace

uint64_t content_hash(const void* data, size_t size) {
    // Four independent lanes (xxHash64-style rounds) so the multiplies overlap.
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t a = kContentPrime1 + kContentPrime2;
    uint64_t b = kContentPrime2;
    uint64_t c = 0;
    uint64_t d = 0 - kContentPrime1;
    size_t i = 0;
    for (; i + 32 <= size; i += 32) {
        a = content_round(a, load_u64(p + i));
        b = content_round(b, load_u64(p + i + 8));
        c = content_round(c, load_u64(p + i + 16));
        d = content_round(d, load_u64(p + i + 24));
    }
    uint64_t h = rotl64(a, 1) + rotl64(b, 7) + rotl64(c, 12) + rotl64(d, 18) + static_cast<uint64_t>(size);
    for (; i + 8 <= size; i += 8) {
        h = rotl64(h ^ content_round(0, load_u64(p + i)), 27) * kContentPrime1 + 0x85EBCA77C2B2AE63ull;
    }
    for (; i < size; ++i) {
        h = rotl64(h ^ (p[i] * 0x27D4EB2F165667C5ull), 11) * kContentPrime1;
    }
    h ^= h >> 33;
    h *= kContentPrime2;
    h ^= h >> 29;
    return h;
}

bool all_zero_bytes(const void* data, size_t size) {
    const auto* p = static_cast<const uint8_t*>(data);
    size_t i = 0;
#if defined(__SSE2__) || defined(_M_X64)
    // 128 bytes per step (eight 16-byte loads OR-ed), one branch per step.
    for (; i + 128 <= size; i += 128) {
        __m128i acc = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i));
        for (size_t k = 16; k < 128; k += 16) {
            acc = _mm_or_si128(acc, _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + i + k)));
        }
        if (_mm_movemask_epi8(_mm_cmpeq_epi8(acc, _mm_setzero_si128())) != 0xFFFF) {
            return false;
        }
    }
#endif
    for (; i + 64 <= size; i += 64) {
        uint64_t acc = 0;
        for (size_t k = 0; k < 64; k += 8) {
            acc |= load_u64(p + i + k);
        }
        if (acc != 0) {
            return false;
        }
    }
    for (; i < size; ++i) {
        if (p[i] != 0) {
            return false;
        }
    }
    return true;
}

uint64_t hash_dirty_bands(const uint8_t* data, uint32_t total_units, uint32_t band_units,
                          size_t unit_stride, size_t unit_bytes, std::vector<uint64_t>* out_bands) {
    if (out_bands != nullptr) {
        out_bands->clear();
    }
    if (data == nullptr || total_units == 0u || band_units == 0u || unit_bytes == 0u || unit_bytes > unit_stride) {
        return 0u;
    }
    const size_t num_bands = (size_t{total_units} + band_units - 1u) / band_units;
    std::vector<uint64_t> local_bands;
    std::vector<uint64_t>& bands = out_bands != nullptr ? *out_bands : local_bands;
    bands.reserve(num_bands);
    for (uint32_t u = 0; u < total_units; u += band_units) {
        const uint32_t n = std::min(band_units, total_units - u);
        if (unit_bytes == unit_stride) {
            bands.push_back(content_hash(data + size_t{u} * unit_stride, size_t{n} * unit_stride));
        } else {
            uint64_t bh = kContentPrime1 + n;
            for (uint32_t k = 0; k < n; ++k) {
                const uint64_t rh = content_hash(data + size_t{u + k} * unit_stride, unit_bytes);
                bh = content_round(bh, rh);
            }
            bands.push_back(bh);
        }
    }
    return content_hash(bands.data(), bands.size() * sizeof(uint64_t));
}

uint32_t compute_dirty_band_runs(const std::vector<uint64_t>& prev_bands,
                                 const std::vector<uint64_t>& cur_bands,
                                 uint32_t total_units,
                                 uint32_t band_units,
                                 const std::vector<uint8_t>* zero_mask,
                                 std::vector<DirtyBandRun>* out_runs) {
    if (out_runs != nullptr) {
        out_runs->clear();
    }
    if (cur_bands.empty() || total_units == 0u || band_units == 0u) {
        return 0u;
    }
    const bool has_prev = prev_bands.size() == cur_bands.size();
    const bool has_zero = !has_prev && zero_mask != nullptr && zero_mask->size() == cur_bands.size();
    uint32_t changed = 0u;
    for (uint32_t b = 0; b < cur_bands.size(); ++b) {
        if (has_prev) {
            if (prev_bands[b] == cur_bands[b]) {
                continue;
            }
        } else if (has_zero && (*zero_mask)[b] != 0u) {
            continue;
        }
        ++changed;
        if (out_runs == nullptr) {
            continue;
        }
        const uint32_t first = b * band_units;
        const uint32_t count = std::min(band_units, total_units - first);
        if (!out_runs->empty() && out_runs->back().first + out_runs->back().count == first) {
            out_runs->back().count += count;
        } else {
            out_runs->push_back(DirtyBandRun{first, count});
        }
    }
    return changed;
}

uint64_t native_stage_identity(const void* code, size_t code_size, const void* metadata, size_t metadata_size) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const auto& [data, size] : {std::pair<const void*, size_t>{code, code_size}, {metadata, metadata_size}}) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i) {
            h ^= p[i];
            h *= 0x00000100000001b3ull;
        }
    }
    return h;
}

// ---------------------------------------------------------------------------------------------
// Device: shaders, VA ranges, tracked buffers, diagnostics
// ---------------------------------------------------------------------------------------------

bool Device::shader_known(uint64_t identity) const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return known_shaders_.count(identity) != 0;
}

void Device::register_shader_module(uint64_t identity, uint32_t module_id) {
    bridge_->register_shader_module(identity, module_id);
    std::lock_guard<std::mutex> lock(shader_mutex_);
    known_shaders_.insert(identity);
    unresolved_shaders_.erase(std::remove(unresolved_shaders_.begin(), unresolved_shaders_.end(), identity),
                              unresolved_shaders_.end());
}

void Device::note_unresolved_shader(uint64_t identity, ShaderStage stage) {
    static constexpr const char* kStageNames[] = {"vertex", "pixel", "compute"};
    const size_t index = static_cast<size_t>(stage);
    std::lock_guard<std::mutex> lock(shader_mutex_);
    if (std::find(unresolved_shaders_.begin(), unresolved_shaders_.end(), identity) == unresolved_shaders_.end()) {
        unresolved_shaders_.push_back(identity);
        ++unresolved_by_stage_[index];
        char name[24];
        std::snprintf(name, sizeof(name), "0x%016" PRIx64, identity);
        auto facts = shader_sizes_.find(identity);
        const int kind = facts != shader_sizes_.end() ? facts->second.kind : -1;
        os::log(std::string("d3d12_xs: no shader module for DXIL identity ") + name + " (" + kStageNames[index] +
                " shader, DXIL kind " + std::to_string(kind) + "; " +
                (stage == ShaderStage::Compute ? "dispatches" : "draws") + " with it are skipped)");
    }
}

std::array<uint32_t, 3> Device::unresolved_shader_stages() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return unresolved_by_stage_;
}

void Device::note_native_stages(uint64_t vertex_identity, uint64_t pixel_identity) {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    const bool first = native_stages_[0].empty();
    if (vertex_identity != 0) {
        native_stages_[0].insert(vertex_identity);
    }
    if (pixel_identity != 0) {
        native_stages_[1].insert(pixel_identity);
    }
    if (first) {
        os::log("d3d12_xs: console-native graphics pipelines (Xbox slot 57): stage code is GPU machine code, "
                "not DXIL; their draws are skipped (distinct stages are counted, not listed)");
    }
}

std::array<uint32_t, 2> Device::native_stage_counts() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return {static_cast<uint32_t>(native_stages_[0].size()), static_cast<uint32_t>(native_stages_[1].size())};
}

std::array<uint32_t, 3> Device::root_signature_versions() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return root_signature_versions_;
}

void Device::note_shader_size(uint64_t identity, uint64_t bytes, int kind) {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    shader_sizes_[identity] = ShaderFacts{bytes, kind};
}

uint64_t Device::shader_size(uint64_t identity) const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    auto it = shader_sizes_.find(identity);
    return it != shader_sizes_.end() ? it->second.bytes : 0;
}

int Device::shader_kind(uint64_t identity) const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    auto it = shader_sizes_.find(identity);
    return it != shader_sizes_.end() ? it->second.kind : -1;
}

bool Device::claim_skipped_draw_report(const void* pipeline_state) {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return described_pipelines_.insert(pipeline_state).second;
}

void Device::add_skipped_draw_report(const std::string& text) {
    os::log(text);
    std::lock_guard<std::mutex> lock(shader_mutex_);
    if (skipped_draw_reports_.size() < 64) {
        skipped_draw_reports_.push_back(text);
    }
}

std::vector<std::string> Device::skipped_draw_reports() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return skipped_draw_reports_;
}

std::vector<uint64_t> Device::unresolved_shaders() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return unresolved_shaders_;
}

uint64_t Device::allocate_gpu_va(uint64_t size) {
    std::lock_guard<std::mutex> lock(va_mutex_);
    const uint64_t va = next_fake_va_;
    next_fake_va_ += align_up(std::max<uint64_t>(size, 1), 64u << 10);
    return va;
}

void Device::track_cpu_buffer(Resource* resource) {
    std::lock_guard<std::mutex> lock(tracked_mutex_);
    tracked_.push_back(resource);
    tracked_unprimed_.insert(resource);
}

void Device::untrack_cpu_buffer(Resource* resource) {
    std::lock_guard<std::mutex> lock(tracked_mutex_);
    tracked_.erase(std::remove(tracked_.begin(), tracked_.end(), resource), tracked_.end());
    tracked_unprimed_.erase(resource);
    slot_index_.remove(resource);
    buffer_page_index_.remove(resource);
    crosscheck_buffers_.erase(resource);
    resource->flush_indexed = false;
    resource->flush_primed = false;
}

namespace {

const void* texture_slot_key(uint32_t resource_id) {
    return reinterpret_cast<const void*>(static_cast<uintptr_t>(resource_id));
}

bool any_window_slot_marked(uint64_t va, uint64_t len) {
    if (!os::guest_window::translated(va) || len == 0) {
        return false;
    }
    constexpr uint64_t kSlotBytes = os::guest_window::kSlotBytes;
    const uint64_t max_va = os::guest_window::kWindowBase + os::guest_window::kWindowMaxSize - 1;
    const uint64_t last_va = len - 1 > max_va - va ? max_va : va + (len - 1);
    const uint64_t last_slot_base = last_va & ~(kSlotBytes - 1);
    for (uint64_t addr = va & ~(kSlotBytes - 1);; addr += kSlotBytes) {
        if (os::guest_window::write_marked(addr)) {
            return true;
        }
        if (addr >= last_slot_base) {
            break;
        }
    }
    return false;
}

}  // namespace

void Device::track_guest_texture(const TrackedTexture& tex) {
    if (tex.resource_id == 0 || tex.gpu_va == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
    texture_slot_index_.remove(texture_slot_key(tex.resource_id));
    TrackedTexture& slot = tracked_textures_[tex.resource_id];
    const TrackedTexture prev = slot;
    slot = tex;
    if (prev.resource_id != 0 && prev.host_floor != 0 && prev.width == tex.width && prev.height == tex.height &&
        prev.mip_levels == tex.mip_levels && prev.dxgi_format == tex.dxgi_format && prev.has_create) {
        // Tracked again after a cap shrink: the host texture is still the shrunk one.
        slot.has_create = true;
        slot.create = prev.create;
        slot.host_floor = prev.host_floor;
        slot.host_saved_bytes = prev.host_saved_bytes;
    }
}

void Device::untrack_guest_texture(uint32_t resource_id) {
    texture_budget().unregister_texture(resource_id);
    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
    texture_slot_index_.remove(texture_slot_key(resource_id));
    if (const auto it = tracked_textures_.find(resource_id); it != tracked_textures_.end()) {
        // A shrunk host texture is counted at its capped size; its owner note_destroyed()s the full size.
        texture_budget().note_shrunk_released(it->second.host_saved_bytes);
        tracked_textures_.erase(it);
    }
    copy_destinations_.erase(resource_id);
    readback_owned_.erase(resource_id);
}

void Device::note_texture_readback_pending(uint32_t resource_id) {
    if (resource_id == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
    readback_owned_.try_emplace(resource_id);
}

void Device::note_texture_readback_retiled(uint32_t resource_id, uint64_t cpu_va, size_t bytes) {
    const void* host = com::guest_to_host(cpu_va, bytes);
    if (resource_id == 0 || host == nullptr) {
        return;
    }
    // The take goes first (it also consumes the re-tile's own writes), so a guest write racing the
    // hash is reported by the next bind's take.
    os::WriteCursor cursor{};
    cursor.peek_watch = true;
    thread_local std::vector<uint64_t> pages;
    pages.clear();
    const bool primed = os::take_written_pages(host, bytes, &cursor, &pages);
    const uint64_t hash = content_hash(host, bytes);
    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
    ReadbackOwned& o = readback_owned_[resource_id];
    o.valid = true;
    o.va = cpu_va;
    o.bytes = bytes;
    o.hash = hash;
    o.cursor = cursor;
    o.cursor_primed = primed;
}

void Device::note_texture_copy_destination(uint32_t resource_id) {
    if (resource_id == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
    texture_slot_index_.remove(texture_slot_key(resource_id));
    copy_destinations_.insert(resource_id);
}

bool Device::tracked_texture_info(uint32_t resource_id, TrackedTexture* out) const {
    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
    const auto it = tracked_textures_.find(resource_id);
    if (it == tracked_textures_.end()) {
        return false;
    }
    if (out != nullptr) {
        *out = it->second;
    }
    return true;
}

void Device::set_detile_fn(DetileFn fn) {
    std::lock_guard<std::mutex> lock(detile_mutex_);
    detile_texture_ = std::move(fn);
    using_default_detile_ = false;
}

namespace {
std::atomic<uint64_t> g_submissions{0};
thread_local uint64_t t_submission = 0;  // 0 = outside ExecuteCommandLists
}  // namespace

void Device::begin_submission() {
    t_submission = g_submissions.fetch_add(1, std::memory_order_relaxed) + 1;
}

void Device::end_submission() {
    t_submission = 0;
}

bool Device::sync_bound_texture_memory(uint32_t resource_id, CommandEncoder& encoder) {
    if (resource_id == 0) {
        return false;
    }
    // Keep actively bound textures at the MRU front of the VRAM budget LRU table even when
    // their contents are unchanged or skipped by write-mark tracking.
    texture_budget().touch_texture(resource_id, present_epoch());
    TrackedTexture tt{};
    // Fast path for the common case, a bound texture whose last full check is still current: its watched
    // host range and cursor are kept in the entry, so a repeat bind only asks the write tracking whether a
    // page was written since (no copy of the entry, no format / layout / host-range work). Taken with a
    // copy of the cursor, which is kept only when nothing was written; otherwise the full path below runs
    // from the old position and sees the same pages.
    const uint8_t* fast_host = nullptr;
    uint64_t fast_size = 0;
    os::WriteCursor fast_cursor{};
    {
        ReadbackOwned owned{};
        bool is_owned = false;
        {
            std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
            const auto ro = readback_owned_.find(resource_id);
            if (ro != readback_owned_.end()) {
                is_owned = true;
                owned = ro->second;
            }
        }
        if (is_owned) {
            if (!owned.valid) {
                return false;  // the first read-back is still in flight
            }
            count_work(Work::TextureSyncOwnedChecks);
            const void* host = com::guest_to_host(owned.va, owned.bytes);
            if (host != nullptr) {
                // No page written since the last take before a matching hash: still the same bytes, no
                // hash. Otherwise hash; on a match the cursor of this take (made before the hash) is kept.
                os::WriteCursor cursor = owned.cursor;
                cursor.peek_watch = true;
                thread_local std::vector<uint64_t> owned_pages;
                owned_pages.clear();
                const bool tracked = os::take_written_pages(host, owned.bytes, &cursor, &owned_pages);
                const bool unwritten = tracked && owned.cursor_primed && owned_pages.empty();
                if (unwritten || content_hash(host, owned.bytes) == owned.hash) {
                    if (unwritten) {
                        count_work(Work::TextureSyncOwnedHashSkips);
                    }
                    std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
                    const auto ro = readback_owned_.find(resource_id);
                    // Only onto the same re-tile's entry (an older cursor would only report more).
                    if (ro != readback_owned_.end() && ro->second.valid && ro->second.hash == owned.hash &&
                        ro->second.va == owned.va && ro->second.bytes == owned.bytes) {
                        ro->second.cursor = cursor;
                        ro->second.cursor_primed = tracked;
                    }
                    return false;  // still the bytes our own re-tile wrote: the GPU copy is current
                }
            }
            // A genuine guest write since the re-tile: the guest owns the texels again.
            std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
            readback_owned_.erase(resource_id);
            if (const auto it = tracked_textures_.find(resource_id); it != tracked_textures_.end()) {
                it->second.window_dirty = true;
                it->second.window_checked = false;
                it->second.view_checked = false;
                it->second.last_upload_hash = 0;
            }
        }
        std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
        if (copy_destinations_.count(resource_id) != 0) {
            return false;  // a recorded copy defines the texels (note_texture_copy_destination)
        }
        const auto it = tracked_textures_.find(resource_id);
        if (it != tracked_textures_.end()) {
            if (t_submission != 0 && it->second.synced_submission == t_submission) {
                return false;  // already synced into an earlier part of this submission
            }
            it->second.synced_submission = t_submission;
            const TrackedTexture& cur = it->second;
            const uint32_t cap_now = texture_budget().current_cap();
            const bool unsettled =
                cur.has_create && cur.layout >= gfx10::kConsoleLinearLayout &&
                cap_floor_for(cur.width, cur.height, std::max(cur.mip_levels, 1u), cap_now) != cur.host_floor;
            if (cur.view_checked && cur.view_host != nullptr && cur.view_size != 0 && cur.view_checked_cap == cap_now &&
                !unsettled && !va_watch::active()) {
                fast_host = cur.view_host;
                fast_size = cur.view_size;
                fast_cursor = cur.view_cursor;
            } else {
                tt = it->second;
            }
        } else {
            bridge::TextureInfo info{};
            if (!bridge_->texture_info(resource_id, &info) || info.swapchain || info.gpu_va == 0 ||
                info.gpu_va >= kFakeGpuVaBase || info.sample_count > 1) {
                return false;
            }
            tt.resource_id = resource_id;
            tt.gpu_va = info.gpu_va;
            tt.dxgi_format = static_cast<uint32_t>(info.format);
            tt.width = info.width;
            tt.height = info.height;
            tt.depth = info.depth_or_array_size;
            tt.mip_levels = info.mip_levels;
            tt.layout = info.layout;
            tt.dimension = info.dimension == GpuTextureDimension::Dim1D
                               ? ResourceDimension::Texture1D
                               : (info.dimension == GpuTextureDimension::Dim3D ? ResourceDimension::Texture3D
                                                                             : ResourceDimension::Texture2D);
            tt.sample_count = info.sample_count;
            tt.synced_submission = t_submission;
            tracked_textures_[resource_id] = tt;
        }
    }
    if (fast_host != nullptr) {
        thread_local std::vector<uint64_t> fast_pages;
        fast_pages.clear();
        fast_cursor.peek_watch = true;  // as the full path's take
        const bool tracked = os::take_written_pages(fast_host, fast_size, &fast_cursor, &fast_pages);
        std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
        const auto it = tracked_textures_.find(resource_id);
        if (it == tracked_textures_.end()) {
            return false;
        }
        TrackedTexture& cur = it->second;
        // Still the range and check the cursor was copied from (another thread's full check may have
        // cleared it meanwhile: then it is not skipped on this cursor).
        if (tracked && fast_pages.empty() && cur.view_checked && cur.view_host == fast_host && cur.view_size == fast_size &&
            cur.view_checked_cap == texture_budget().current_cap()) {
            cur.view_cursor = fast_cursor;
            count_work(Work::TextureSyncViewTracked);
            count_work(Work::TextureSyncViewSkips);
            return false;
        }
        tt = cur;  // written (or could not tell): the full path, from the entry's own cursor
    }
    if (tt.gpu_va == 0 || tt.width == 0 || tt.height == 0 || tt.sample_count > 1) {
        return false;
    }
    const DxgiFormatLayout l = dxgi_format_layout(tt.dxgi_format);
    if (!l.known || l.block_bytes == 0 || l.block_width == 0 || l.block_height == 0 || l.planes > 1) {
        return false;
    }
    bridge::TextureInfo created{};
    const bool have_info = bridge_->texture_info(resource_id, &created);
    const pso::FormatUsage map_usage =
        have_info && created.depth_stencil  ? pso::FormatUsage::DepthStencil
        : have_info && created.render_target ? pso::FormatUsage::RenderTarget
        : have_info && created.sampled_only  ? pso::FormatUsage::SampledOnly
                                             : pso::FormatUsage::Sampled;
    const pso::TextureFormatMapping mapping =
        pso::map_texture_format(static_cast<pso::DxgiFormat>(tt.dxgi_format), map_usage, bridge_->options().features);
    if (!mapping.supported() || pso::has_depth_aspect(mapping.format) || pso::has_stencil_aspect(mapping.format)) {
        return false;
    }
    const bool format_mismatch =
        have_info && created.created_format != pso::WgpuTextureFormat::Undefined &&
        !bridge::is_compatible_view_format(created.created_format, mapping.format);
    const bool needs_conversion =
        format_mismatch ||
        (mapping.diagnostics & (pso::FormatDiag::NeedsCpuConversion | pso::FormatDiag::FeatureFallback)) != 0;
    // 16-bit normalized texels for a SampledOnly texture: converted to its float representation
    // (pso::convert_norm16_texels; exact as float32, reported once when it is float16).
    uint8_t probe_dst[4] = {};
    const uint8_t probe_src[2] = {};
    const bool convert_norm16 = !format_mismatch && needs_conversion &&
                                pso::convert_norm16_texels(static_cast<pso::DxgiFormat>(tt.dxgi_format),
                                                           mapping.format, probe_src, 1, probe_dst);
    if (convert_norm16 && (mapping.diagnostics & pso::FormatDiag::LossyConversion) != 0) {
        bool first = false;
        {
            std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
            first = lossy_reported_.insert(tt.dxgi_format).second;
        }
        if (first) {
            os::log(std::string("d3d12_xs: DXGI format ") + std::to_string(tt.dxgi_format) +
                    " is sampled as " + pso::texture_format_name(mapping.format) +
                    " (16-bit normalized formats are not filterable in WebGPU and float32-filterable is not "
                    "enabled): uploads round each value to float16 (11-bit mantissa)");
        }
    }
    // Promoted CPU-upload formats (A8_UNORM, B5G5R5A1_UNORM, B4G4R4A4_UNORM -> rgba8unorm;
    // Xbox HDR R10G10B10_7E3_A2_FLOAT / R10G10B10_6E4_A2_FLOAT -> rgba16float).
    const CpuWidenSpec widen_spec =
        (!format_mismatch && needs_conversion)
            ? cpu_widen_spec(static_cast<pso::DxgiFormat>(tt.dxgi_format), mapping.format)
            : CpuWidenSpec{};
    const bool convert_widen = widen_spec.src_bpp != 0;
    if (needs_conversion && !convert_norm16 && !convert_widen) {
        // The GPU texture has a substitute format (e.g. rgba8unorm for BC without the feature):
        // the guest's texels do not fit it and are not converted. Skip, reported once per texture.
        bool first = false;
        {
            std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
            first = conversion_skipped_.insert(resource_id).second;
        }
        if (first) {
            note_unsupported_copy("texture upload: DXGI format " + std::to_string(tt.dxgi_format) +
                                  " has a substitute WebGPU format that needs CPU conversion (not implemented); "
                                  "texels not uploaded");
        }
        return false;
    }
    const uint32_t w = tt.width;
    const uint32_t h = (tt.dimension == ResourceDimension::Texture1D) ? 1u : std::max<uint32_t>(1u, tt.height);
    const uint32_t d = (tt.dimension == ResourceDimension::Texture3D) ? std::max<uint32_t>(1u, tt.depth) : 1u;
    const uint32_t layers = (tt.dimension == ResourceDimension::Texture3D) ? 1u : std::max<uint32_t>(1u, tt.depth);
    const uint32_t mips = std::max<uint32_t>(1u, tt.mip_levels);
    const uint32_t blocks_x = (w + l.block_width - 1u) / l.block_width;
    const uint32_t blocks_y = (h + l.block_height - 1u) / l.block_height;
    const uint64_t tight_row_bytes = uint64_t{blocks_x} * l.block_bytes;
    const uint64_t tight_mip0_bytes = tight_row_bytes * blocks_y * d;
    if (va_watch::active()) {  // (the detail string is built on every bind otherwise)
        va_watch::touch("upload (bound texture memory)", tt.gpu_va, tight_mip0_bytes * layers,
                        "texture id " + std::to_string(resource_id) + ", DXGI format " + std::to_string(tt.dxgi_format));
    }
    constexpr uint64_t kMaxSyncTextureBytes = 64u << 20;
    if (tight_mip0_bytes == 0 || align_up(tight_row_bytes, 256u) * blocks_y * d > kMaxSyncTextureBytes) {
        return false;
    }

    // Console layouts (0x100 | swizzle mode, linear 0x100 included) are placed by the GFX10
    // surface code through the detile hook; D3D12 layouts 0-3 are read as tight rows.
    const bool is_tiled = tt.layout >= gfx10::kConsoleLinearLayout;
    const bool is_tiny = (w <= 2u && h <= 2u);
    DetileFn detile_fn;
    bool using_default_detile = false;
    {
        std::lock_guard<std::mutex> lock(detile_mutex_);
        detile_fn = detile_texture_;
        using_default_detile = using_default_detile_;
    }
    if (is_tiled && !detile_fn && !is_tiny) {
        return false;
    }

    // The subresources this layout places (subresource index = mip + slice * mips). Console
    // layouts: every mip and array slice of the GFX10 surface (gfx10::MipPlacement; linear 0x100
    // follows addrlib's ADDR_SW_LINEAR chain, #678). D3D12 layouts 0-3: mip 0 at the base; array
    // slices only where every candidate rule gives the same slice stride (one mip, rows a
    // multiple of 256 B).
    struct Sub {
        uint32_t mip;
        uint32_t slice;
    };
    std::vector<Sub> subs;
    uint64_t tiled_bytes = 0;
    if (is_tiled && detile_fn) {
        tiled_bytes = tiled_surface_bytes(tt.layout, tt.dxgi_format, w, h, d, mips, layers);
    }
    const bool whole_surface = tiled_bytes != 0;
    const uint32_t linear_layers = (!is_tiled && layers > 1u && mips == 1u && tight_row_bytes % 256u == 0u) ? layers : 1u;
    for (uint32_t s = 0; s < (whole_surface ? layers : linear_layers); ++s) {
        for (uint32_t m = 0; m < (whole_surface ? mips : 1u); ++m) {
            subs.push_back({m, s});
        }
    }

    // Safely access the source bytes via CPU-window translation (never crashes on unmapped/guard pages).
    const uint64_t cpu_va = cpu_address_for_gpu(tt.gpu_va);
    const uint64_t fallback_size = tight_mip0_bytes * linear_layers;
    uint64_t primary_size = fallback_size;
    if (is_tiled && detile_fn) {
        if (tiled_bytes == 0) {
            GuestResourceDesc tmp{};
            tmp.Dimension = tt.dimension;
            tmp.Width = w;
            tmp.Height = h;
            tmp.DepthOrArraySize = static_cast<uint16_t>(std::min<uint32_t>(d, 0xFFFFu));
            tmp.MipLevels = 1;
            tmp.Format = tt.dxgi_format;
            tmp.SampleCount = 1;
            tmp.Layout = tt.layout;
            tiled_bytes = committed_memory_size(tmp);
        }
        const uint64_t tiled_read = std::max<uint64_t>(tiled_bytes, tight_mip0_bytes);
        if (tiled_read > 0 && tiled_read <= kMaxSyncTextureBytes) {
            primary_size = tiled_read;
        }
    }

    // G68 (#663): when the texture lives in the translated CPU window, skip reading, zero-scanning
    // and hashing altogether if none of its 64 KiB slots were marked written since the last check
    // (either via flush_cpu_buffers's shared MarkPass or directly since then).
    // A host-reallocatable texture whose allocation does not match the cap in force is due a pass even
    // when its memory is unchanged (the cap moved, or a restore was held back by the hysteresis).
    const auto host_unsettled = [&](const TrackedTexture& cur) {
        return cur.has_create && cur.layout >= gfx10::kConsoleLinearLayout &&
               cap_floor_for(cur.width, cur.height, std::max(cur.mip_levels, 1u), texture_budget().current_cap()) !=
                   cur.host_floor;
    };
    const bool in_window = os::guest_window::translated(cpu_va);
    const bool verify_marks = in_window && os::guest_window::write_marks_verified();
    bool unmarked = false;
    bool needs_window_checked = false;
    if (in_window && primary_size > 0) {
        std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
        const auto it = tracked_textures_.find(resource_id);
        if (it == tracked_textures_.end()) {
            return false;
        }
        TrackedTexture& cur = it->second;
        const bool same_range = cur.window_indexed && cur.window_indexed_va == cpu_va &&
                                cur.window_indexed_size == primary_size;
        const bool dirty_now = !same_range || cur.window_dirty || any_window_slot_marked(cpu_va, primary_size) ||
                               host_unsettled(cur);
        if (!same_range) {
            texture_slot_index_.add(texture_slot_key(resource_id), cpu_va, primary_size);
            cur.window_indexed = true;
            cur.window_indexed_va = cpu_va;
            cur.window_indexed_size = primary_size;
            cur.window_checked = false;
            cur.streaming_full_frames = 0;
        }
        if (!dirty_now && cur.window_checked) {
            unmarked = true;
            if (!verify_marks) {
                cur.streaming_full_frames = 0;
            }
        } else {
            // Clear before reading so a write racing with the scan re-marks the slot for the next sync.
            cur.window_dirty = false;
        }
        needs_window_checked = !cur.window_checked;
    }
    if (unmarked && !verify_marks) {
        return false;
    }

    // Native shared page views (no translated window): the same skip, from the per-page write
    // stamps of os::take_written_pages (the OS write bits are fanned out to every cursor; false where the
    // range cannot be tracked, which keeps the full read). A texture whose last full check
    // (zero scan / hash) is still current and whose pages were not written since is unchanged.
    bool view_tracked = false;
    if (!in_window && primary_size > 0) {
        if (const auto* host = static_cast<const uint8_t*>(com::guest_to_host(cpu_va, static_cast<size_t>(primary_size)))) {
            os::WriteCursor cursor{};
            bool same_range = false;
            {
                std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
                const auto it = tracked_textures_.find(resource_id);
                if (it == tracked_textures_.end()) {
                    return false;
                }
                same_range = it->second.view_host == host && it->second.view_size == primary_size;
                if (same_range) {
                    cursor = it->second.view_cursor;
                }
            }
            std::vector<uint64_t> pages;
            cursor.peek_watch = true;  // uncollected OS write bits count as written; no reset per texture
            const bool tracked = os::take_written_pages(host, primary_size, &cursor, &pages);
            bool skip = false;
            {
                std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
                const auto it = tracked_textures_.find(resource_id);
                if (it == tracked_textures_.end()) {
                    return false;
                }
                TrackedTexture& cur = it->second;
                // a changed texture-budget cap (#1596) is never skipped: dropped levels may be due
                const uint32_t cap_now = texture_budget().current_cap();
                skip = tracked && same_range && cur.view_checked && pages.empty() && cur.view_checked_cap == cap_now &&
                       !host_unsettled(cur);
                if (!skip) {
                    cur.view_checked = false;  // set again once a full check of this call completes
                }
                cur.view_host = host;
                cur.view_size = primary_size;
                cur.view_cursor = cursor;  // also after a refusal: its commit position is current, never stale
            }
            count_work(tracked ? Work::TextureSyncViewTracked : Work::TextureSyncUntracked);
            if (skip) {
                count_work(Work::TextureSyncViewSkips);
                return false;
            }
            view_tracked = tracked;
        }
    }
    auto note_view_checked = [&]() {
        if (!view_tracked) {
            return;
        }
        std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
        if (auto it = tracked_textures_.find(resource_id); it != tracked_textures_.end()) {
            it->second.view_checked = true;
            it->second.view_checked_cap = texture_budget().current_cap();
        }
    };

    auto note_window_checked = [&]() {
        note_view_checked();
        if (!needs_window_checked) {
            return;
        }
        std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
        if (auto it = tracked_textures_.find(resource_id); it != tracked_textures_.end()) {
            if (it->second.window_indexed && it->second.window_indexed_va == cpu_va &&
                it->second.window_indexed_size == primary_size) {
                it->second.window_checked = true;
            }
        }
    };

    // G68 (#663): inspect contiguous guest memory in place (zero allocation / copy when unchanged),
    // falling back to a segmented com::read_guest copy only when window slots are non-adjacent.
    std::vector<uint8_t> raw;
    std::optional<com::GuestRangeView> view;  // arena lock held while src_data points into guest memory
    const uint8_t* src_data = nullptr;
    size_t src_size = 0;
    auto try_map_or_read = [&](uint64_t size) -> bool {
        if (size == 0) {
            return false;
        }
        view.emplace(cpu_va, static_cast<size_t>(size));
        if (const uint8_t* live = view->data()) {
            src_data = live;
            src_size = static_cast<size_t>(size);
            return true;
        }
        view.reset();
        raw.resize(static_cast<size_t>(size));
        if (com::read_guest(cpu_va, raw.data(), raw.size())) {
            src_data = raw.data();
            src_size = raw.size();
            return true;
        }
        raw.clear();
        return false;
    };
    bool read_ok = try_map_or_read(primary_size);
    if (!read_ok && fallback_size != primary_size) {
        read_ok = try_map_or_read(fallback_size);
    }
    if (!read_ok) {
        return false;
    }

    if (!tt.ever_uploaded && all_zero_bytes(src_data, src_size)) {
        note_window_checked();
        return false;
    }

    // G72 (#663): For a single tightly-packed linear surface (whether or not tight_row_bytes is
    // 256B-aligned, e.g., 4K Bink2 Y/U/V planes), hash its 64-row bands once in place over tight
    // rows and derive the surface hash from them so the surface is never hashed twice or eagerly
    // copied/zero-filled into an intermediate staging buffer.
    // G73 (#663): For a single-subresource console-linear surface (layout == 0x100, mips=1,
    // layers=1, d=1) using the built-in GFX10 detiler, addrlib ADDR_SW_LINEAR already places mip 0
    // at offset 0 with 256B-aligned row pitch (`aligned_row_pitch * blocks_y` bytes). Borrow
    // `src_data` directly with single-pass band hashing and skip detile + staging copy.
    const uint64_t aligned_row_pitch = align_up(tight_row_bytes, 256u);
    const uint64_t console_linear_bytes = aligned_row_pitch * blocks_y;
    const bool single_linear_direct =
        subs.size() == 1u && !is_tiled && !convert_norm16 && !convert_widen && src_size == tight_mip0_bytes;
    const bool single_console_linear_direct =
        using_default_detile && tt.layout == gfx10::kConsoleLinearLayout && whole_surface &&
        subs.size() == 1u && mips == 1u && layers == 1u && d == 1u && !convert_norm16 && !convert_widen &&
        tiled_bytes == console_linear_bytes && src_size >= console_linear_bytes;
    // G75 (#663): Streaming full-frame textures (e.g. 4K Bink2 Y/U/V planes) modify >= 75% of
    // bands every dirty frame. After 3 consecutive full-surface dirty updates in a write-marked
    // translated window, skip per-band content_hash() while dirty marks keep arriving.
    const bool streaming_eligible =
        in_window && !verify_marks && (single_console_linear_direct || single_linear_direct) &&
        d == 1u && blocks_y > kUploadBandRows;
    const bool skip_streaming_hash =
        streaming_eligible && tt.ever_uploaded && tt.streaming_full_frames >= 3u;
    std::vector<uint64_t> direct_bands;
    uint64_t hash = 0;
    if (skip_streaming_hash) {
        // Streaming fast path: skip content_hash() and emit full surface directly.
    } else if (single_linear_direct && d == 1u && blocks_y > kUploadBandRows) {
        direct_bands.reserve((blocks_y + kUploadBandRows - 1u) / kUploadBandRows);
        for (uint32_t r = 0; r < blocks_y; r += kUploadBandRows) {
            const uint32_t n = std::min(kUploadBandRows, blocks_y - r);
            direct_bands.push_back(
                content_hash(src_data + size_t{r} * tight_row_bytes, size_t{n} * tight_row_bytes));
        }
        hash = content_hash(direct_bands.data(), direct_bands.size() * sizeof(uint64_t));
    } else if (single_console_linear_direct && blocks_y > kUploadBandRows) {
        direct_bands.reserve((blocks_y + kUploadBandRows - 1u) / kUploadBandRows);
        for (uint32_t r = 0; r < blocks_y; r += kUploadBandRows) {
            const uint32_t n = std::min(kUploadBandRows, blocks_y - r);
            direct_bands.push_back(
                content_hash(src_data + size_t{r} * aligned_row_pitch, size_t{n} * aligned_row_pitch));
        }
        hash = content_hash(direct_bands.data(), direct_bands.size() * sizeof(uint64_t));
    } else if (single_console_linear_direct) {
        hash = content_hash(src_data, static_cast<size_t>(console_linear_bytes));
    } else {
        hash = content_hash(src_data, src_size);
    }
    if (!skip_streaming_hash && tt.ever_uploaded && tt.last_upload_hash != 0 && hash == tt.last_upload_hash &&
        !host_unsettled(tt)) {
        if (streaming_eligible && tt.streaming_full_frames != 0) {
            std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
            if (auto it = tracked_textures_.find(resource_id); it != tracked_textures_.end()) {
                it->second.streaming_full_frames = 0;
            }
        }
        note_window_checked();
        return false;
    }
    if (unmarked && verify_marks && !any_window_slot_marked(cpu_va, primary_size)) {
        os::guest_window::note_verify_miss(cpu_va);
    }

    // Past the check: the staging below runs long, so take a private copy and release the arena lock
    // (a concurrent decommit must not fault the detile / upload reads).
    if (view && view->held()) {
        raw.assign(src_data, src_data + src_size);
        src_data = raw.data();
    }
    view.reset();

    // Stage every placed subresource; only those whose texels changed since their last upload
    // are written (per-subresource dirty tracking), so one changed mip costs one upload. Inside a
    // 2D subresource only the changed bands of kUploadBandRows block rows are written (#691: a
    // glyph atlas the title fills a few glyphs at a time re-uploaded 16 MiB per new glyph); inside
    // a 3D subresource only the changed depth slices are written (#1747).
    uint32_t cap_floor = 0;
    const bool cap_eligible =
        whole_surface && tt.dimension == ResourceDimension::Texture2D && l.block_width > 1u && mips > 1u;
    if (cap_eligible) {
        cap_floor = cap_floor_for(w, h, mips, texture_budget().current_cap());
    }
    gfx10::TextureLayoutDesc surface_desc{};
    surface_desc.dxgi_format = tt.dxgi_format;
    surface_desc.width = w;
    surface_desc.height = h;
    surface_desc.depth = d;
    surface_desc.mip_levels = mips;
    surface_desc.d3d12_layout = tt.layout;
    surface_desc.array_size = d > 1u ? 1u : layers;

    struct Staged {
        Sub sub;
        uint32_t copy_w, copy_h, copy_d, row_pitch, rows;
        uint64_t hash;
        std::vector<uint8_t> bytes;         // owned staging bytes (empty when borrowing src_data)
        const uint8_t* borrowed = nullptr;  // non-null when referencing src_data directly
        size_t borrowed_size = 0;
        uint32_t src_row_bytes = 0;         // tight row stride when needs_row_pad is true
        bool needs_row_pad = false;         // true: borrowed has tight rows, pad to row_pitch on emit
        bool band_is_z = false;             // true: bands/runs index 3D depth slices; false: 2D row bands
        std::vector<uint64_t> bands;        // hash per band (2D rows) or per depth slice (3D)
        std::vector<DirtyBandRun> runs;     // units [first, first + count) to write

        const uint8_t* data() const { return bytes.empty() ? borrowed : bytes.data(); }
        size_t size() const { return bytes.empty() ? borrowed_size : bytes.size(); }
    };
    std::vector<Staged> staged;
    staged.reserve(subs.size());
    for (const Sub& sub : subs) {
        const size_t sub_idx = sub.mip + size_t{sub.slice} * mips;
        const uint64_t prev_sub_hash = sub_idx < tt.subresource_hashes.size() ? tt.subresource_hashes[sub_idx] : 0u;
        // Skip detiling, allocating, converting, and hashing top mips that will be dropped by the
        // current texture size cap (cap_floor can only stay equal or increase under hysteresis).
        if (sub.mip < cap_floor && prev_sub_hash == 0u) {
            continue;
        }
        // Fast path for unstreamed standard (non-tail) console-layout 2D/array/cubemap mips:
        // inspect the contiguous tiled/console-linear macro-block span in place before detiling.
        if (using_default_detile && is_tiled && whole_surface && d == 1u && prev_sub_hash == 0u &&
            !skip_streaming_hash && !single_console_linear_direct) {
            const gfx10::MipPlacement mp = gfx10::mip_placement(surface_desc, sub.mip);
            if (mp.ok && !mp.in_tail && mp.slice_bytes > 0u && layers > 0u) {
                const uint64_t slice_stride = mp.surface_bytes / layers;
                const uint64_t mip_off = slice_stride * sub.slice + mp.macro_block_offset;
                if (mip_off <= src_size && mp.slice_bytes <= src_size - mip_off &&
                    all_zero_bytes(src_data + mip_off, static_cast<size_t>(mp.slice_bytes))) {
                    continue;
                }
            }
        }
        const uint32_t mw = std::max(1u, w >> sub.mip);
        const uint32_t mh = std::max(1u, h >> sub.mip);
        const uint32_t md = std::max(1u, d >> sub.mip);
        const uint32_t bx = (mw + l.block_width - 1u) / l.block_width;
        const uint32_t by = (mh + l.block_height - 1u) / l.block_height;
        const uint64_t row_bytes = uint64_t{bx} * l.block_bytes;
        const uint64_t row_pitch = align_up(row_bytes, 256u);
        Staged st{sub, bx * l.block_width, by * l.block_height, md, static_cast<uint32_t>(row_pitch), by, 0, {},
                  nullptr, 0, 0, false, false, {}, {}};
        if (skip_streaming_hash) {
            if (single_console_linear_direct) {
                st.borrowed = src_data;
                st.borrowed_size = static_cast<size_t>(console_linear_bytes);
                st.row_pitch = static_cast<uint32_t>(aligned_row_pitch);
            } else {
                st.borrowed = src_data;
                st.borrowed_size = src_size;
                if (row_pitch != row_bytes) {
                    st.src_row_bytes = static_cast<uint32_t>(row_bytes);
                    st.needs_row_pad = true;
                }
            }
            st.runs.push_back(DirtyBandRun{0u, by});
            staged.push_back(std::move(st));
            continue;
        }
        if (single_console_linear_direct) {
            st.borrowed = src_data;
            st.borrowed_size = static_cast<size_t>(console_linear_bytes);
            st.row_pitch = static_cast<uint32_t>(aligned_row_pitch);
            st.bands = std::move(direct_bands);
            st.hash = hash;
            staged.push_back(std::move(st));
            continue;
        }
        bool detiled = false;
        if (is_tiled && detile_fn) {
            st.bytes.assign(static_cast<size_t>(row_pitch * by * md), 0);
            TextureDetileRequest req{};
            req.dxgi_format = tt.dxgi_format;
            req.width = w;
            req.height = h;
            req.depth = d;
            req.mip_levels = mips;
            req.layout = tt.layout;
            req.block_bytes = l.block_bytes;
            req.block_width = l.block_width;
            req.block_height = l.block_height;
            req.dst_row_pitch = st.row_pitch;
            req.array_size = whole_surface ? layers : 1u;
            req.mip = sub.mip;
            req.array_slice = sub.slice;
            detiled = detile_fn(req, src_data, src_size, st.bytes.data(), st.bytes.size());
            if (!detiled && (sub.mip != 0u || sub.slice != 0u)) {
                continue;  // the hook does not place this subresource
            }
            if (!detiled && !is_tiny) {
                return false;
            }
            if (!detiled) {
                st.bytes.clear();
            }
        }
        const uint8_t* linear_src = nullptr;
        uint64_t linear_src_pitch = row_pitch;
        if (!detiled) {
            const uint64_t slice_off = uint64_t{sub.slice} * tight_mip0_bytes;
            if (convert_norm16 || convert_widen) {
                linear_src = src_data + static_cast<size_t>(slice_off);
                linear_src_pitch = row_bytes;
            } else if (row_pitch == row_bytes) {
                st.borrowed = src_data + static_cast<size_t>(slice_off);
                st.borrowed_size = static_cast<size_t>(row_pitch * by * md);
            } else if (single_linear_direct) {
                // Lazy row-pad: borrow tight src_data and repack only changed runs during emit.
                st.borrowed = src_data;
                st.borrowed_size = src_size;
                st.src_row_bytes = static_cast<uint32_t>(row_bytes);
                st.needs_row_pad = true;
            } else {
                const uint32_t total_rows = by * md;
                st.bytes.assign(static_cast<size_t>(row_pitch * total_rows), 0);
                for (uint32_t r = 0; r < total_rows; ++r) {
                    std::memcpy(st.bytes.data() + static_cast<size_t>(r) * row_pitch,
                                src_data + static_cast<size_t>(slice_off + r * row_bytes),
                                static_cast<size_t>(row_bytes));
                }
            }
        } else {
            linear_src = st.bytes.data();
            linear_src_pitch = row_pitch;
        }
        if (convert_norm16) {
            // Widen each row of 16-bit normalized components into the texture's float format.
            const size_t comps = static_cast<size_t>(row_bytes / 2u);
            const size_t out_bpc = mapping.format == pso::WgpuTextureFormat::R32Float ||
                                           mapping.format == pso::WgpuTextureFormat::Rg32Float ||
                                           mapping.format == pso::WgpuTextureFormat::Rgba32Float
                                       ? 4u
                                       : 2u;
            const uint64_t out_pitch = align_up(comps * out_bpc, 256u);
            std::vector<uint8_t> converted(static_cast<size_t>(out_pitch * by * md), 0);
            for (uint32_t r = 0; r < by * md; ++r) {
                pso::convert_norm16_texels(static_cast<pso::DxgiFormat>(tt.dxgi_format), mapping.format,
                                           linear_src + static_cast<size_t>(r) * linear_src_pitch, comps,
                                           converted.data() + static_cast<size_t>(r) * out_pitch);
            }
            st.bytes = std::move(converted);
            st.row_pitch = static_cast<uint32_t>(out_pitch);
        }
        if (convert_widen) {
            const size_t texels_per_row = static_cast<size_t>(row_bytes / widen_spec.src_bpp);
            const uint64_t out_pitch = align_up(uint64_t{texels_per_row} * widen_spec.dst_bpp, 256u);
            std::vector<uint8_t> converted(static_cast<size_t>(out_pitch * by * md), 0);
            for (uint32_t r = 0; r < by * md; ++r) {
                widen_cpu_texels(static_cast<pso::DxgiFormat>(tt.dxgi_format), mapping.format,
                                 linear_src + static_cast<size_t>(r) * linear_src_pitch, texels_per_row,
                                 converted.data() + static_cast<size_t>(r) * out_pitch);
            }
            st.bytes = std::move(converted);
            st.row_pitch = static_cast<uint32_t>(out_pitch);
        }
        if (single_linear_direct && !direct_bands.empty()) {
            st.bands = std::move(direct_bands);
            st.hash = hash;
        } else if (single_linear_direct && st.copy_d == 1u && !(st.rows > kUploadBandRows)) {
            st.hash = hash;
        } else if (st.copy_d > 1u) {
            // 3D subresource: per-depth-slice dirty tracking so sparse 3D volume updates only upload
            // the modified Z slices (#1747).
            st.band_is_z = true;
            const size_t active_row_stride = st.needs_row_pad ? size_t{st.src_row_bytes} : size_t{st.row_pitch};
            const size_t slice_stride = size_t{st.rows} * active_row_stride;
            st.hash = hash_dirty_bands(st.data(), st.copy_d, /*band_units=*/1u, slice_stride, slice_stride, &st.bands);
        } else if (st.copy_d == 1u && st.rows > kUploadBandRows) {
            const size_t active_row_stride = st.needs_row_pad ? size_t{st.src_row_bytes} : size_t{st.row_pitch};
            st.hash = hash_dirty_bands(st.data(), st.rows, kUploadBandRows, active_row_stride, active_row_stride,
                                       &st.bands);
        } else {
            st.hash = content_hash(st.data(), st.size());
        }
        staged.push_back(std::move(st));
    }

    uint32_t held_floor = 0;
    // Host reallocation (see gpu/texture_budget.h): set below when the host texture is re-created.
    bool host_recreate = false;
    GpuCreateTexture2Payload recreate{};
    uint32_t host_floor_now = 0;
    int64_t saved_delta = 0;
    uint64_t host_new_bytes = 0;
    const bool single_known_nonzero =
        !tt.ever_uploaded &&
        (single_linear_direct || (single_console_linear_direct && src_size == console_linear_bytes));
    {
        std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
        const auto it = tracked_textures_.find(resource_id);
        if (it == tracked_textures_.end()) {
            return false;
        }
        TrackedTexture& cur = it->second;
        if (cur.has_create && cur.host_floor != 0 && !cap_eligible) {
            cap_floor = cur.host_floor;  // a partial pass cannot refill a shrunk host texture: keep it
        }
        if (cur.has_create && cap_eligible && cap_floor != cur.host_floor) {
            bool change = true;
            const auto chain = [&](uint32_t floor) {
                return texture_chain_bytes(w, h, 1u, mips, l.block_width, l.block_height, l.block_bytes, floor, layers);
            };
            if (cap_floor < cur.host_floor) {
                // Hysteresis: grow back only while the larger texture still fits the tier its cap came from.
                const uint64_t extra = chain(cap_floor) - chain(cur.host_floor);
                change = texture_budget().restore_fits(w, h, mips, cap_floor, extra);
            }
            if (change) {
                const uint64_t new_chain = chain(cap_floor);
                const uint64_t saved = chain(0) - new_chain;
                saved_delta = static_cast<int64_t>(saved) - static_cast<int64_t>(cur.host_saved_bytes);
                host_new_bytes = new_chain;
                cur.host_saved_bytes = saved;
                cur.host_floor = cap_floor;
                recreate = cur.create;
                recreate.width = std::max(1u, w >> cap_floor);
                recreate.height = std::max(1u, h >> cap_floor);
                recreate.mip_levels = mips - cap_floor;
                host_recreate = true;
                // The new allocation is empty: every held level is uploaded again.
                std::fill(cur.subresource_hashes.begin(), cur.subresource_hashes.end(), 0u);
                for (auto& b : cur.subresource_bands) {
                    b.clear();
                }
                cur.last_upload_hash = 0;
            } else {
                cap_floor = cur.host_floor;
            }
        }
        host_floor_now = cur.has_create ? cur.host_floor : 0u;
        // Backlog 37: the size cap loosened since the last pass, so dropped top levels are due again.
        const bool cap_restore = cap_floor < cur.cap_floor_applied;
        const bool was_ever_uploaded = cur.ever_uploaded;
        if (!skip_streaming_hash && cur.ever_uploaded && cur.last_upload_hash != 0 && cur.last_upload_hash == hash &&
            !cap_restore) {
            if (streaming_eligible) {
                cur.streaming_full_frames = 0;
            }
            if (view_tracked) {
                cur.view_checked = true;
                cur.view_checked_cap = texture_budget().current_cap();
            }
            if (needs_window_checked && cur.window_indexed && cur.window_indexed_va == cpu_va &&
                cur.window_indexed_size == primary_size) {
                cur.window_checked = true;
            }
            return false;
        }
        cur.last_upload_hash = skip_streaming_hash ? 0 : hash;
        cur.ever_uploaded = true;
        if (view_tracked) {
            cur.view_checked = true;
            cur.view_checked_cap = texture_budget().current_cap();
        }
        if (needs_window_checked && cur.window_indexed && cur.window_indexed_va == cpu_va &&
            cur.window_indexed_size == primary_size) {
            cur.window_checked = true;
        }
        cur.subresource_hashes.resize(size_t{mips} * layers, 0);
        cur.subresource_bands.resize(size_t{mips} * layers);
        if (cap_restore) {
            texture_budget().note_restored();
        }
        const uint32_t prev_cap_floor = cur.cap_floor_applied;
        cur.cap_floor_applied = cap_floor;
        auto out = staged.begin();
        for (Staged& st : staged) {
            const size_t index = st.sub.mip + size_t{st.sub.slice} * mips;
            uint64_t& seen = cur.subresource_hashes[index];
            if (st.sub.mip < cap_floor && seen == 0) {
                continue;  // dropped top level under memory pressure: lower resolution, never uploaded
            }
            std::vector<uint64_t>& prev = cur.subresource_bands[index];
            if (skip_streaming_hash) {
                prev.clear();
                seen = 0;
                if (st.runs.empty()) {
                    st.runs.push_back(DirtyBandRun{0u, st.band_is_z ? st.copy_d : st.rows});
                }
                if (&*out != &st) {
                    *out = std::move(st);
                }
                ++out;
                continue;
            }
            if (seen == st.hash) {
                if (streaming_eligible) {
                    cur.streaming_full_frames = 0;
                }
                continue;  // unchanged
            }
            if (seen == 0 && !(was_ever_uploaded && subs.size() == 1u) && !single_known_nonzero &&
                all_zero_bytes(st.data(), st.size())) {
                continue;  // still zero as the GPU texture was created
            }
            const uint32_t total_units = st.band_is_z ? st.copy_d : st.rows;
            const uint32_t band_units = st.band_is_z ? 1u : kUploadBandRows;
            if (!st.bands.empty()) {
                if (seen != 0 && prev.size() == st.bands.size()) {
                    const uint32_t changed_bands = compute_dirty_band_runs(prev, st.bands, total_units, band_units,
                                                                           /*zero_mask=*/nullptr, &st.runs);
                    if (streaming_eligible && !st.band_is_z) {
                        if (changed_bands * 4u >= st.bands.size() * 3u) {
                            cur.streaming_full_frames =
                                static_cast<uint8_t>(std::min<uint32_t>(uint32_t{cur.streaming_full_frames} + 1u, 8u));
                        } else {
                            cur.streaming_full_frames = 0;
                        }
                    }
                } else if (st.band_is_z && seen == 0 && (!was_ever_uploaded || host_recreate)) {
                    // Initial upload of a freshly zero-initialized 3D GPU texture: skip all-zero Z slices
                    // while still recording every slice hash in `prev` so future writes trigger uploads.
                    std::vector<uint8_t> zero_mask(st.bands.size(), 0u);
                    const size_t active_row_stride =
                        st.needs_row_pad ? size_t{st.src_row_bytes} : size_t{st.row_pitch};
                    const size_t unit_stride = size_t{st.rows} * active_row_stride;
                    const uint8_t* ptr = st.data();
                    for (uint32_t b = 0; b < st.bands.size(); ++b) {
                        if (all_zero_bytes(ptr + size_t{b} * unit_stride, unit_stride)) {
                            zero_mask[b] = 1u;
                        }
                    }
                    static const std::vector<uint64_t> kEmptyPrev;
                    compute_dirty_band_runs(kEmptyPrev, st.bands, total_units, band_units, &zero_mask, &st.runs);
                }
            }
            if (st.runs.empty()) {
                st.runs.push_back(DirtyBandRun{0u, total_units});
            }
            prev = st.bands;
            seen = st.hash;
            if (&*out != &st) {
                *out = std::move(st);
            }
            ++out;
        }
        staged.erase(out, staged.end());
        held_floor = bridge::held_mip_floor(cur.subresource_hashes, mips, layers, cur.streamable);
        if (cap_floor != 0) {
            held_floor = std::max(held_floor, cap_floor);
            if (cap_floor > prev_cap_floor) {
                texture_budget().note_capped(
                    cap_floor - prev_cap_floor,
                    texture_chain_bytes(w, h, 1u, cap_floor, l.block_width, l.block_height, l.block_bytes,
                                        prev_cap_floor, layers));
            }
        }
    }
    if (host_recreate) {
        replace_host_texture(recreate, encoder);
        bridge_->set_host_mip_offset(resource_id, host_floor_now);
        texture_budget().note_host_realloc(saved_delta);
        texture_budget().update_texture_bytes(resource_id, host_new_bytes);
    }
    // G66 partly-held mip chains: views of the texture start at its first held level.
    bridge_->set_resident_mip_floor(resource_id, held_floor);

    struct PadScratch {
        std::unique_ptr<uint8_t[]> data;
        size_t capacity = 0;

        uint8_t* ensure(size_t bytes) {
            if (bytes > capacity) {
                data.reset(new uint8_t[bytes]);
                capacity = bytes;
            }
            return data.get();
        }

        void release() {
            data.reset();
            capacity = 0;
        }
    };
    constexpr size_t kMaxRetainedPadScratch = 16u << 20;
    thread_local PadScratch pad_scratch;
    bool used_pad_scratch = false;

    for (const Staged& st : staged) {
        if (st.sub.mip < host_floor_now) {
            continue;  // a level the capped host texture lacks
        }
        for (const DirtyBandRun& run : st.runs) {
            const uint32_t first = run.first;
            const uint32_t count = run.count;
            const uint8_t* data = nullptr;
            uint32_t len = 0;
            if (st.band_is_z) {
                if (st.needs_row_pad) {
                    const size_t total_rows = size_t{st.rows} * count;
                    const size_t padded_bytes = total_rows * st.row_pitch;
                    const size_t src_row_bytes = st.src_row_bytes;
                    const size_t pad_tail = st.row_pitch - src_row_bytes;
                    uint8_t* scratch = pad_scratch.ensure(padded_bytes);
                    const uint8_t* src_base = st.borrowed + size_t{first} * st.rows * src_row_bytes;
                    uint8_t* dst = scratch;
                    for (size_t r = 0; r < total_rows; ++r) {
                        std::memcpy(dst, src_base, src_row_bytes);
                        std::memset(dst + src_row_bytes, 0, pad_tail);
                        src_base += src_row_bytes;
                        dst += st.row_pitch;
                    }
                    data = scratch;
                    len = static_cast<uint32_t>(padded_bytes);
                    used_pad_scratch = true;
                } else {
                    const size_t slice_bytes = size_t{st.rows} * st.row_pitch;
                    data = st.data() + size_t{first} * slice_bytes;
                    len = static_cast<uint32_t>(size_t{count} * slice_bytes);
                }
                encoder.write_texture(resource_id, st.sub.mip - host_floor_now, 0u, 0u, first, st.copy_w, st.copy_h,
                                      count, st.row_pitch, st.rows, data, len);
            } else {
                const uint32_t bh = st.copy_h / st.rows;  // block height (copy_h is whole blocks)
                if (st.needs_row_pad) {
                    const size_t total_rows = size_t{count} * st.copy_d;
                    const size_t padded_bytes = total_rows * st.row_pitch;
                    const size_t src_row_bytes = st.src_row_bytes;
                    const size_t pad_tail = st.row_pitch - src_row_bytes;
                    uint8_t* scratch = pad_scratch.ensure(padded_bytes);
                    const uint8_t* src_base = st.borrowed + size_t{first} * src_row_bytes;
                    uint8_t* dst = scratch;
                    for (size_t r = 0; r < total_rows; ++r) {
                        std::memcpy(dst, src_base, src_row_bytes);
                        std::memset(dst + src_row_bytes, 0, pad_tail);
                        src_base += src_row_bytes;
                        dst += st.row_pitch;
                    }
                    data = scratch;
                    len = static_cast<uint32_t>(padded_bytes);
                    used_pad_scratch = true;
                } else {
                    data = st.data() + size_t{first} * st.row_pitch;
                    len = static_cast<uint32_t>(
                        count == st.rows ? st.size() : size_t{count} * st.row_pitch);
                }
                encoder.write_texture(resource_id, st.sub.mip - host_floor_now, 0u, first * bh, st.sub.slice,
                                      st.copy_w, count * bh, st.copy_d, st.row_pitch, count, data, len);
            }
            count_work(Work::TextureUploads);
            note_upload(len);
            note_texture_upload();
        }
    }
    if (used_pad_scratch && pad_scratch.capacity > kMaxRetainedPadScratch) {
        pad_scratch.release();
    }
    return !staged.empty();
}

void Device::note_address_texture_evicted(const AddressTextureKey& key) {
    if (evicted_address_keys_.size() >= kMaxEvictedAddressKeys) {
        evicted_address_keys_.clear();
    }
    evicted_address_keys_.insert(AddressTextureKeyHash{}(key));
    small_texture_pool().note_address_eviction();
}

uint32_t Device::ensure_address_texture(uint64_t gpu_va, uint32_t dxgi_format, uint32_t width, uint32_t height,
                                        uint32_t depth, uint32_t mip_levels, uint32_t d3d12_layout,
                                        GpuTextureDimension dim, bool uav) {
    if (gpu_va == 0 || width < 1u || height < 1u || width > 16384u || height > 16384u) {
        return 0;
    }
    const uint32_t safe_height = dim == GpuTextureDimension::Dim1D ? 1u : height;
    const uint32_t safe_depth = std::max<uint32_t>(1u, depth);
    if (safe_depth > 2048u) {
        return 0;
    }
    uint32_t max_mips = 1u;
    uint32_t extent = std::max<uint32_t>(width, safe_height);
    if (dim == GpuTextureDimension::Dim3D) {
        extent = std::max<uint32_t>(extent, safe_depth);
    }
    while (extent > 1u) {
        extent >>= 1u;
        ++max_mips;
    }
    const uint32_t safe_mips = mip_levels == 0u ? max_mips : std::min<uint32_t>(mip_levels, max_mips);

    // A texture known only by a view is never a render target: SampledOnly.
    const pso::TextureFormatMapping mapping = pso::map_texture_format(
        static_cast<pso::DxgiFormat>(dxgi_format), pso::FormatUsage::SampledOnly, bridge_->options().features);
    if (!mapping.supported() || pso::has_depth_aspect(mapping.format) || pso::has_stencil_aspect(mapping.format)) {
        return 0;
    }

    AddressTextureKey key{};
    key.gpu_va = gpu_va;
    key.dxgi_format = dxgi_format;
    key.width = width;
    key.height = safe_height;
    key.depth = safe_depth;
    key.mip_levels = safe_mips;
    key.d3d12_layout = d3d12_layout;
    key.dim = static_cast<uint8_t>(dim);

    const DxgiFormatLayout shape_layout = dxgi_format_layout(dxgi_format);
    const DxgiFormatLayout host_layout = wgpu_format_block_layout(mapping.format);
    const DxgiFormatLayout& effective_layout =
        (host_layout.known && host_layout.block_bytes != 0) ? host_layout : shape_layout;
    const bool is3d_tex = dim == GpuTextureDimension::Dim3D;
    const uint64_t chain_bytes =
        effective_layout.known && effective_layout.block_bytes != 0
            ? texture_chain_bytes(width, safe_height, is3d_tex ? safe_depth : 1u, safe_mips,
                                  effective_layout.block_width, effective_layout.block_height,
                                  effective_layout.block_bytes, 0u, is3d_tex ? 1u : safe_depth)
            : 0u;
    const uint64_t cur_frame = present_epoch();

    std::lock_guard<std::mutex> lock(address_textures_mutex_);
    const auto it = address_textures_.find(key);
    if (it != address_textures_.end()) {
        it->second.last_use = ++address_texture_tick_;
        texture_budget().touch_texture(it->second.id, cur_frame);
        return it->second.id;
    }
    // O36 / G70 (#691) / G82 (#1747): VRAM-budget LRU reclamation + bounded count LRU.
    // Reclaim cold address textures when VRAM pressure crosses the hysteresis trigger threshold,
    // then enforce the entry count ceiling. Prefer evicting textures not used in the current frame,
    // and defer DestroyResource until after Present for any in-frame victim.
    CommandEncoder drop;
    bridge_->advance_present_frame(presents(), drop);
    // Parked large textures stay counted in the budget: give them back before the use nears the
    // size-cap tiers or LRU reclamation, so they never cost a live texture its top levels.
    if (const uint64_t parked_large = large_texture_pool().stats().parked_bytes; parked_large != 0) {
        const TextureBudgetStats tb = texture_budget().stats();
        const uint64_t limit = tb.budget_bytes / 100u * kLargeTexturePoolBudgetPercent;
        if (tb.budget_bytes != 0 && tb.used_bytes + chain_bytes > limit) {
            const uint64_t over = tb.used_bytes + chain_bytes - limit;
            std::vector<uint32_t> ids;
            std::vector<uint64_t> bytes;
            large_texture_pool().trim(UINT32_MAX, parked_large - std::min(parked_large, over), &ids, &bytes);
            for (size_t i = 0; i < ids.size(); ++i) {
                texture_budget().note_destroyed(bytes[i]);
                bridge_->unregister_resource(ids[i]);
                bridge_->destroy_or_defer_resource(ids[i], drop);
            }
        }
    }
    if (const uint64_t reclaim_goal = texture_budget().bytes_to_reclaim(chain_bytes); reclaim_goal != 0u) {
        const std::vector<LruEvictCandidate> candidates =
            texture_budget().select_eviction_candidates(cur_frame, reclaim_goal);
        for (const LruEvictCandidate& cand : candidates) {
            if (bridge_->is_resource_used_in_frame(cand.resource_id)) {
                continue;
            }
            auto victim_it = address_textures_.end();
            for (auto i = address_textures_.begin(); i != address_textures_.end(); ++i) {
                if (i->second.id == cand.resource_id) {
                    victim_it = i;
                    break;
                }
            }
            if (victim_it == address_textures_.end()) {
                continue;
            }
            const uint32_t victim = victim_it->second.id;
            const uint64_t victim_bytes = victim_it->second.budget_bytes;
            note_address_texture_evicted(victim_it->first);
            address_textures_.erase(victim_it);
            untrack_guest_texture(victim);
            bridge_->unregister_resource(victim);
            if (victim_bytes != 0) {
                texture_budget().note_destroyed(victim_bytes);
            }
            texture_budget().note_lru_evicted(victim, victim_bytes != 0 ? victim_bytes : cand.bytes);
            bridge_->destroy_or_defer_resource(victim, drop);
        }
    }
    if (address_textures_.size() >= kMaxAddressTextures) {
        // Drop the least recently used down to kMaxAddressTextures - kAddressTextureEvictBatch in one
        // pass, preferring textures the frame being recorded does not reference.
        struct Candidate {
            decltype(address_textures_)::iterator it;
            bool in_frame;
            uint64_t use;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(address_textures_.size());
        {
            const bridge::StateBridgeDevice::ResourceUseScan use_scan(*bridge_);
            for (auto i = address_textures_.begin(); i != address_textures_.end(); ++i) {
                candidates.push_back(Candidate{i, use_scan.in_frame(i->second.id),
                                               std::max(i->second.last_use, use_scan.last_use_tick(i->second.id))});
            }
        }
        const size_t keep = kMaxAddressTextures - std::min(kAddressTextureEvictBatch, kMaxAddressTextures);
        const size_t n_evict = std::min(candidates.size(), candidates.size() - std::min(candidates.size(), keep) + 1u);
        const auto older = [](const Candidate& a, const Candidate& b) {
            return a.in_frame != b.in_frame ? !a.in_frame : a.use < b.use;
        };
        std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(n_evict),
                          candidates.end(), older);
        struct Victim {
            uint32_t id;
            AddressTextureEntry entry;
            bool in_frame;
            bool shrunk;  // host texture re-created at a capped size: not the shape it is keyed by
        };
        std::vector<Victim> victims;
        std::vector<uint32_t> victim_ids;
        victims.reserve(n_evict);
        victim_ids.reserve(n_evict);
        for (size_t v = 0; v < n_evict; ++v) {
            const auto oldest = candidates[v].it;
            Victim victim{oldest->second.id, oldest->second, candidates[v].in_frame, false};
            {
                std::lock_guard<std::mutex> tracked_lock(tracked_textures_mutex_);
                const auto t = tracked_textures_.find(victim.id);
                victim.shrunk =
                    t != tracked_textures_.end() && (t->second.host_floor != 0 || t->second.host_saved_bytes != 0);
            }
            note_address_texture_evicted(oldest->first);
            address_textures_.erase(oldest);
            untrack_guest_texture(victim.id);
            victims.push_back(victim);
            victim_ids.push_back(victim.id);
        }
        bridge_->unregister_resources(victim_ids);
        std::vector<uint32_t> pool_evicted;
        std::vector<uint64_t> pool_evicted_bytes;
        for (const Victim& victim : victims) {
            // Backlog 38: a texture not referenced by the frame being recorded is parked for reuse
            // instead of destroyed (it stays counted in the texture budget while parked). Chains over
            // 64 KiB go to the large pool, only when the consumer zeroes reused textures on the GPU.
            if (!victim.in_frame && victim.entry.budget_bytes != 0 && !victim.shrunk &&
                texture_pool_for(victim.entry.budget_bytes)
                    .release(victim.entry.shape, victim.id, victim.entry.budget_bytes, &pool_evicted,
                             &pool_evicted_bytes)) {
                continue;
            }
            if (victim.entry.budget_bytes != 0) {
                texture_budget().note_destroyed(victim.entry.budget_bytes);
                texture_budget().note_lru_evicted(victim.id, victim.entry.budget_bytes);
            }
            bridge_->destroy_or_defer_resource(victim.id, drop);
        }
        if (!pool_evicted.empty()) {
            for (size_t i = 0; i < pool_evicted.size(); ++i) {
                texture_budget().note_destroyed(pool_evicted_bytes[i]);
                untrack_guest_texture(pool_evicted[i]);
            }
            bridge_->unregister_resources(pool_evicted);
            for (const uint32_t id : pool_evicted) {
                bridge_->destroy_or_defer_resource(id, drop);
            }
        }
    }
    if (drop.size() != 0) {
        submit(drop);
    }
    small_texture_pool().note_address_miss(evicted_address_keys_.erase(AddressTextureKeyHash{}(key)) != 0);

    // Backlog 38: the host texture of a released texture of the same shape is reused; every texel of
    // every mip is zeroed below (on the GPU when the consumer can), so nothing of its previous
    // content stays visible.
    const uint32_t features = bridge_->options().features;
    const pso::WgpuTextureFormat create_format = storage_base_format(mapping.format, uav, 1u, features);
    const bool can_store = uav && storage_capable(create_format, features);
    const bool can_render_uav =
        uav && !can_store && dim == GpuTextureDimension::Dim2D && renderable_uav_fallback(create_format, features);
    SmallTextureShape shape{};
    shape.owner = reinterpret_cast<uintptr_t>(this);
    shape.dxgi_format = dxgi_format;
    shape.width = width;
    shape.height = safe_height;
    shape.depth = safe_depth;
    shape.mip_levels = safe_mips;
    shape.dim = static_cast<uint8_t>(dim);
    shape.storage = can_store;
    const uint32_t reused_id = can_render_uav ? 0u : texture_pool_for(chain_bytes).acquire(shape, chain_bytes);
    const uint32_t id = reused_id != 0 ? reused_id : allocate_id();
    GpuCreateTexture2Payload payload{};
    payload.id = id;
    payload.width = width;
    payload.height = safe_height;
    payload.depth_or_layers = safe_depth;
    payload.mip_levels = safe_mips;
    payload.sample_count = 1;
    payload.dimension = static_cast<uint8_t>(dim);
    payload.usage_flags = GpuTextureUsage::TextureBinding | GpuTextureUsage::CopyDst;
    if (can_store) {
        payload.usage_flags |= GpuTextureUsage::StorageBinding;
    } else if (can_render_uav) {
        payload.usage_flags |= GpuTextureUsage::RenderAttachment;
    }
    payload.format = static_cast<uint8_t>(create_format);
    if (const pso::WgpuTextureFormat sibling = pso::srgb_sibling(create_format);
        sibling != pso::WgpuTextureFormat::Undefined) {
        payload.view_format_count = 1;  // other descriptors may view it sRGB / linear
        payload.view_formats[0] = static_cast<uint8_t>(sibling);
    }

    CommandEncoder enc;
    if (reused_id != 0 && gpu_texture_clear()) {
        GpuClearTexturePayload clear{};
        clear.texture_id = id;
        enc.emit_packet(kGpuOpcodeClearTexture, &clear, sizeof(clear));
    } else if (reused_id != 0) {
        const std::vector<ZeroWrite> zeros =
            zero_fill_writes(width, safe_height, safe_depth, safe_mips, is3d_tex, effective_layout.block_width,
                             effective_layout.block_height, effective_layout.block_bytes);
        std::vector<uint8_t> zero_bytes;
        for (const ZeroWrite& z : zeros) {
            zero_bytes.assign(static_cast<size_t>(z.bytes), 0u);
            enc.write_texture(id, z.mip, 0, 0, 0, z.width, z.height, z.depth_or_layers, z.bytes_per_row,
                              z.rows_per_image, zero_bytes.data(), static_cast<uint32_t>(zero_bytes.size()));
        }
    } else {
        const auto t0 = std::chrono::steady_clock::now();
        enc.emit_packet(GpuOpcode::CreateTexture2, &payload, sizeof(payload));
        small_texture_pool().note_create(
            chain_bytes, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() - t0)
                                                   .count()));
        if (chain_bytes != 0) {
            texture_budget().note_created(chain_bytes);
        }
    }
    submit(enc);

    bridge::TextureInfo info{};
    info.format = static_cast<pso::DxgiFormat>(dxgi_format);
    info.width = width;
    info.height = safe_height;
    info.depth_or_array_size = safe_depth;
    info.mip_levels = safe_mips;
    info.sample_count = 1;
    info.dimension = dim;
    info.gpu_va = gpu_va;
    info.layout = d3d12_layout;
    info.sampled_only = true;
    info.storage = (payload.usage_flags & GpuTextureUsage::StorageBinding) != 0;
    info.render_target = can_render_uav;
    info.created_format = create_format;
    bridge_->register_texture(id, info);

    if (gpu_va < kFakeGpuVaBase) {
        TrackedTexture tt{};
        tt.resource_id = id;
        tt.gpu_va = gpu_va;
        tt.dxgi_format = dxgi_format;
        tt.width = width;
        tt.height = safe_height;
        tt.depth = safe_depth;
        tt.mip_levels = safe_mips;
        tt.layout = d3d12_layout;
        tt.dimension = dim == GpuTextureDimension::Dim1D
                           ? ResourceDimension::Texture1D
                           : (dim == GpuTextureDimension::Dim3D ? ResourceDimension::Texture3D
                                                                : ResourceDimension::Texture2D);
        tt.sample_count = 1;
        if (dim == GpuTextureDimension::Dim2D && safe_depth == 1u && safe_mips > 1u && shape_layout.known &&
            shape_layout.block_width > 1u && chain_bytes > kSmallTextureMaxBytes &&
            (payload.usage_flags & (GpuTextureUsage::RenderAttachment | GpuTextureUsage::StorageBinding)) == 0) {
            tt.has_create = true;  // the host allocation may be replaced at a capped size
            tt.create = payload;
        }
        track_guest_texture(tt);
        // "Textures with data" counter: the surface's bytes (console layouts), else the tight mip 0.
        const bool is3d = dim == GpuTextureDimension::Dim3D;
        uint64_t size = tiled_surface_bytes(d3d12_layout, dxgi_format, width, safe_height, is3d ? safe_depth : 1u,
                                            safe_mips, is3d ? 1u : safe_depth);
        if (size == 0) {
            const DxgiFormatLayout l = dxgi_format_layout(dxgi_format);
            size = l.known && l.block_width != 0 && l.block_height != 0
                       ? uint64_t{(width + l.block_width - 1u) / l.block_width} *
                             ((safe_height + l.block_height - 1u) / l.block_height) * l.block_bytes
                       : 0u;
        }
        if (size != 0) {
            register_view_texture_probe(cpu_address_for_gpu(gpu_va), size);
        }
    }

    address_textures_.emplace(key, AddressTextureEntry{id, ++address_texture_tick_, chain_bytes, shape});
    texture_budget().register_texture(id, chain_bytes, cur_frame, /*pinned=*/false, /*evictable=*/true);
    return id;
}

namespace {
std::atomic<uint8_t> g_flush_mode{static_cast<uint8_t>(FlushMode::Index)};
std::atomic<uint32_t> g_crosscheck_every{1};
}  // namespace

void set_flush_mode(FlushMode mode) {
    g_flush_mode.store(static_cast<uint8_t>(mode), std::memory_order_relaxed);
}

FlushMode flush_mode() {
    return static_cast<FlushMode>(g_flush_mode.load(std::memory_order_relaxed));
}

void set_flush_crosscheck_every(uint32_t n) {
    g_crosscheck_every.store(n == 0 ? 1u : n, std::memory_order_relaxed);
}

void Device::flush_cpu_buffers(CommandEncoder& encoder) {
    os::lock_with_span(flush_mutex_, os::SpanKind::D3dLock);
    std::lock_guard<std::mutex> flush_lock(flush_mutex_, std::adopt_lock);
    // Odd while a flush runs: bumped before any write mark is taken and again when it ends.
    flush_epoch_.fetch_add(1, std::memory_order_acq_rel);
    struct FlushEpochEnd {
        std::atomic<uint64_t>& epoch;
        ~FlushEpochEnd() { epoch.fetch_add(1, std::memory_order_acq_rel); }
    } flush_epoch_end{flush_epoch_};
    // G70 (#691): emit deferred DestroyResource packets from the previous frame once Present has
    // advanced (strictly after that frame's Present and before any draw in the new frame).
    bridge_->advance_present_frame(presents(), encoder);
    // O36: one pass over the translated window's write marks for the buffers and the relocated
    // ranges below. A relocated range in the window whose slots are unmarked has not been written
    // since its bytes were last read (the same protocol as the buffers: marks taken before the
    // read), so it is not read and compared again.
    mark_pass_.begin();
    auto unwritten = [&](uint64_t cpu, uint64_t size) {
        flush_marked_.clear();
        return mark_pass_.take(cpu, static_cast<size_t>(size), &flush_marked_) && flush_marked_.empty();
    };
    // Native arena ranges (outside the translated window): the OS write tracking says whether any page was
    // written since this range's previous take. The take re-arms the tracking before the caller hashes,
    // so a write racing with the hash is seen next time. The first take (or one the platform cannot
    // answer) reports "possibly written" and only primes the cursor.
    auto native_unwritten = [&](os::WriteCursor& cursor, bool& primed, const void* host, uint64_t size) {
        flush_native_written_.clear();
        const bool ok = os::take_written_pages(host, size, &cursor, &flush_native_written_);
        const bool skip = ok && primed && flush_native_written_.empty();
        primed = ok;
        return skip;
    };
    // O36: take every dirty window slot once, then visit only the buffers and tracked textures
    // overlapping one (sharing this submission's MarkPass so a shared 64 KiB slot notifies both).
    flush_dirty_slots_.clear();
    const bool scanned = mark_pass_.take_dirty(&flush_dirty_slots_);
    // #1963: native flushes (no translated window) are driven by the OS write tracking. The host
    // ranges of the indexed buffers and relocated ranges (buffer_page_index_, uniform_page_index_,
    // storage_page_index_, ia_page_index_) are taken once here, with the flush's own cursor; only
    // the entries overlapping a written page are flushed / compared, plus every entry not primed yet
    // (new, re-keyed, or not covered by a successful take at its last visit) and every entry of a
    // range whose take could not tell.
    // Why that is enough: a primed entry was compared after the take of a flush covering it, so every
    // write since lands after that take and is reported by a later one, which flushes the entry.
    // Buffers the tracking does not cover (no take of their own succeeds) are flushed every time.
    const FlushMode mode = flush_mode();
    const bool indexed = !scanned && !os::guest_window::write_marks_verified() && mode != FlushMode::Full;
    const uint64_t seq = ++flush_seq_;
    const bool crosscheck = indexed && mode == FlushMode::CrossCheck &&
                            seq % g_crosscheck_every.load(std::memory_order_relaxed) == 0;
    // The indexed flushes' own cost (crosscheck flushes excluded), for A/B numbers in one run.
    struct IndexedTimer {
        bool on;
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~IndexedTimer() {
            if (on) {
                count_work(Work::FlushIndexedCalls);
                count_work(Work::FlushIndexedNs, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                         std::chrono::steady_clock::now() - t0)
                                                                         .count()));
            }
        }
    } indexed_timer{indexed && !crosscheck};
    if (indexed != page_index_live_) {
        // Entering or leaving the indexed mode: start over (every entry unprimed and re-indexed).
        {
            std::lock_guard<std::mutex> lock(tracked_mutex_);
            buffer_page_index_.clear();
            crosscheck_buffers_.clear();
            tracked_unprimed_.clear();
            for (Resource* r : tracked_) {
                if (r->flush_primed && r->flush_visit_seq + 1 < seq) {
                    // Skipped on the previous indexed flush(es): flush_cursor_ is their take end.
                    r->advance_write_cursor(flush_cursor_);
                }
                r->flush_indexed = false;
                r->flush_primed = false;
                tracked_unprimed_.insert(r);
            }
        }
        {
            std::lock_guard<std::mutex> lock(relocation_mutex_);
            uniform_page_index_.clear();
            crosscheck_uniforms_.clear();
            uniform_unprimed_.clear();
            for (auto& [key, r] : relocations_) {
                r.primed = false;
                uniform_unprimed_.insert(&r);
            }
        }
        {
            std::lock_guard<std::mutex> lock(storage_relocation_mutex_);
            storage_page_index_.clear();
            crosscheck_storage_.clear();
            storage_unprimed_.clear();
            for (auto& [key, r] : storage_relocations_) {
                r.index_primed = false;
                storage_unprimed_.insert(&r);
            }
        }
        {
            std::lock_guard<std::mutex> lock(ia_relocation_mutex_);
            ia_page_index_.clear();
            crosscheck_ia_.clear();
            ia_unprimed_.clear();
            for (auto& [key, r] : ia_relocations_) {
                r.index_primed = false;
                ia_unprimed_.insert(&r);
            }
        }
        page_index_live_ = indexed;
    }
    // The ranges taken are the indexed buffers' and relocated ranges' (merged, page-aligned).
    flush_take_ranges_.clear();
    flush_taken_ok_.clear();
    flush_taken_failed_.clear();
    flush_dirty_pages_.clear();
    if (indexed) {
        std::lock_guard<std::mutex> lock(tracked_mutex_);
        const auto& h = buffer_page_index_.hulls();
        flush_take_ranges_.assign(h.begin(), h.end());
    }
    if (indexed) {
        // The relocated uniform ranges are taken too (also where no buffer covers them), so only the
        // ones overlapping a written page are compared.
        std::lock_guard<std::mutex> lock(relocation_mutex_);
        const auto& h = uniform_page_index_.hulls();
        flush_take_ranges_.insert(flush_take_ranges_.end(), h.begin(), h.end());
    }
    if (indexed) {
        // And the relocated storage ranges, so only the ones overlapping a written page are hashed.
        std::lock_guard<std::mutex> lock(storage_relocation_mutex_);
        const auto& h = storage_page_index_.hulls();
        flush_take_ranges_.insert(flush_take_ranges_.end(), h.begin(), h.end());
    }
    if (indexed) {
        // And the relocated IA (vertex / index) ranges.
        std::lock_guard<std::mutex> lock(ia_relocation_mutex_);
        const auto& h = ia_page_index_.hulls();
        flush_take_ranges_.insert(flush_take_ranges_.end(), h.begin(), h.end());
        merge_ranges(&flush_take_ranges_);
    }
    const auto buffers_t0 = std::chrono::steady_clock::now();
    // The OS write bits of the buffers and of the relocated storage / index ranges, collected in one pass
    // that outlives the loops below (each take inside the ranges then skips its own GetWriteWatch).
    flush_host_ranges_.clear();
    std::optional<os::WriteBatch> write_batch;
    {
        std::lock_guard<std::mutex> lock(tracked_mutex_);
        // Verification mode compares unmarked pages too, so it visits everything.
        if (!scanned || os::guest_window::write_marks_verified()) {
            slot_index_.clear();
            // Native: every tracked buffer takes its written pages below. Collect the OS write bits
            // of all of them in one pass first, so each buffer does not pay a GetWriteWatch call.
            // (Indexed entries lie inside flush_take_ranges_.)
            flush_host_ranges_.reserve(flush_take_ranges_.size() + (indexed ? tracked_unprimed_.size() + 64u : tracked_.size()));
            flush_host_ranges_.insert(flush_host_ranges_.end(), flush_take_ranges_.begin(), flush_take_ranges_.end());
            const auto add_buffer_range = [&](const Resource* r) {
                if (r->cpu_address() == 0 || r->shadow_bytes() == 0) {
                    return;
                }
                if (const auto* live = static_cast<const uint8_t*>(
                        com::guest_to_host(r->cpu_address(), static_cast<size_t>(r->shadow_bytes())))) {
                    flush_host_ranges_.emplace_back(reinterpret_cast<uintptr_t>(live),
                                                    reinterpret_cast<uintptr_t>(live) + static_cast<uintptr_t>(r->shadow_bytes()));
                }
            };
            if (indexed) {
                // Every non-indexed buffer has flush_primed == false and is in tracked_unprimed_.
                for (const Resource* r : tracked_unprimed_) {
                    if (!r->flush_indexed) {
                        add_buffer_range(r);
                    }
                }
            } else {
                for (const Resource* r : tracked_) {
                    add_buffer_range(r);
                }
            }
            const auto add_range = [&](uint64_t cpu, uint64_t size) {
                if (const auto* live = static_cast<const uint8_t*>(com::guest_to_host(cpu, static_cast<size_t>(size)))) {
                    flush_host_ranges_.emplace_back(reinterpret_cast<uintptr_t>(live),
                                                    reinterpret_cast<uintptr_t>(live) + static_cast<uintptr_t>(size));
                }
            };
            {
                std::lock_guard<std::mutex> rlock(storage_relocation_mutex_);
                if (indexed) {
                    // Every entry outside storage_page_index_ has index_primed == false (in storage_unprimed_).
                    for (const RelocatedStorage* r : storage_unprimed_) {
                        if (!storage_page_index_.contains(r)) {
                            add_range(cpu_address_for_gpu(r->va), r->size);
                        }
                    }
                } else {
                    for (const auto& [key, r] : storage_relocations_) {
                        add_range(cpu_address_for_gpu(r.va), r.size);
                    }
                }
            }
            {
                std::lock_guard<std::mutex> rlock(ia_relocation_mutex_);
                if (indexed) {
                    // Every entry outside ia_page_index_ has index_primed == false (in ia_unprimed_).
                    for (const RelocatedIa* r : ia_unprimed_) {
                        if (!ia_page_index_.contains(r)) {
                            add_range(cpu_address_for_gpu(r->va), r->size);
                        }
                    }
                } else {
                    for (const auto& [key, r] : ia_relocations_) {
                        add_range(cpu_address_for_gpu(r.va), r.size);
                    }
                }
            }
            write_batch.emplace(&flush_host_ranges_);
            if (indexed) {
                // One take over all ranges from the flush cursor (os::take_written_pages_ranges).
                // Save `from` (the previous flush's take end) so any primed buffer skipped on prior
                // flushes can lazily advance its own write cursor to `from` when it is next visited.
                const os::WriteCursor from = flush_cursor_;
                flush_take_ok_flags_.clear();
                os::take_written_pages_ranges(flush_take_ranges_, &flush_cursor_, &flush_dirty_pages_,
                                              &flush_take_ok_flags_);
                for (size_t i = 0; i < flush_take_ranges_.size(); ++i) {
                    (flush_take_ok_flags_[i] != 0 ? flush_taken_ok_ : flush_taken_failed_)
                        .push_back(flush_take_ranges_[i]);
                }
                std::sort(flush_dirty_pages_.begin(), flush_dirty_pages_.end());
                flush_dirty_pages_.erase(std::unique(flush_dirty_pages_.begin(), flush_dirty_pages_.end()),
                                         flush_dirty_pages_.end());
                count_work(Work::FlushIndexTakes, flush_take_ranges_.size());
                count_work(Work::FlushIndexTakeFailures, flush_taken_failed_.size());
                count_work(Work::FlushIndexDirtyPages, flush_dirty_pages_.size());

                // The buffers the takes name (primed ones; the unprimed ones are visited below too).
                flush_visit_keys_.clear();
                buffer_page_index_.collect_pages(flush_dirty_pages_, os::kWrittenPageSize, &flush_visit_keys_);
                for (const auto& h : flush_taken_failed_) {
                    buffer_page_index_.collect_range(h.first, h.second, &flush_visit_keys_);
                }
                for (const void* key : flush_visit_keys_) {
                    Resource* r = const_cast<Resource*>(static_cast<const Resource*>(key));
                    if (r->flush_primed && r->flush_visit_seq != seq && r->flush_visit_seq + 1 < seq) {
                        // Skipped on the previous indexed flush(es): no page of it was written before
                        // `from`, so its own take in flush_shadow may start at `from`.
                        r->advance_write_cursor(from);
                    }
                    r->flush_visit_seq = seq;
                }
                // --flush-crosscheck: an entry skipped last time although it changed must be named now.
                for (const void* key : crosscheck_buffers_) {
                    const Resource* r = static_cast<const Resource*>(key);
                    if (r->flush_visit_seq == seq || !r->flush_primed) {
                        count_work(Work::FlushCrosscheckLate);
                    } else {
                        count_work(Work::FlushCrosscheckMisses);
                        std::fprintf(stderr, "[flush-crosscheck] buffer %u (guest 0x%llx, %llu bytes) changed unreported\n",
                                     r->id(), static_cast<unsigned long long>(r->cpu_address()),
                                     static_cast<unsigned long long>(r->shadow_bytes()));
                    }
                }
                crosscheck_buffers_.clear();
                uint64_t visited = 0, skipped = 0, untracked = 0;
                auto visit_buffer = [&](Resource* r) {
                    if (r->cpu_address() == 0 || r->shadow_bytes() == 0) {
                        return;
                    }
                    const bool skip = r->flush_primed && r->flush_visit_seq != seq;
                    if (skip && !crosscheck) {
                        ++skipped;
                        return;
                    }
                    if (r->flush_primed && r->flush_visit_seq != seq && r->flush_visit_seq + 1 < seq) {
                        // Skipped on the previous indexed flush(es) and visited now only for crosscheck.
                        r->advance_write_cursor(from);
                    }
                    r->set_slot_indexed(false);
                    const uint64_t uploaded = r->flush_shadow(encoder, &mark_pass_);
                    r->flush_visit_seq = seq;
                    ++visited;
                    if (skip) {
                        ++skipped;
                        if (uploaded != 0) {
                            crosscheck_buffers_.insert(r);
                        }
                    }
                    uintptr_t lo = 0, hi = 0;
                    if (r->write_tracked_host_range(&lo, &hi)) {
                        buffer_page_index_.add(r, lo, hi);
                        r->flush_indexed = true;
                        r->flush_primed = ranges_cover(flush_taken_ok_, lo, hi);
                    } else {
                        buffer_page_index_.remove(r);
                        r->flush_indexed = false;
                        r->flush_primed = false;
                        ++untracked;
                    }
                    if (r->flush_primed) {
                        tracked_unprimed_.erase(r);
                    } else {
                        tracked_unprimed_.insert(r);
                    }
                };
                if (!crosscheck) {
                    // Only the buffers a take reported (flush_visit_seq == seq) and the unprimed ones
                    // need visiting: O(dirty + unprimed) instead of O(tracked_).
                    flush_buffer_candidates_.clear();
                    flush_buffer_candidates_.reserve(flush_visit_keys_.size() + tracked_unprimed_.size());
                    for (const void* key : flush_visit_keys_) {
                        flush_buffer_candidates_.push_back(const_cast<Resource*>(static_cast<const Resource*>(key)));
                    }
                    flush_buffer_candidates_.insert(flush_buffer_candidates_.end(), tracked_unprimed_.begin(),
                                                    tracked_unprimed_.end());
                    std::sort(flush_buffer_candidates_.begin(), flush_buffer_candidates_.end(),
                              [](const Resource* a, const Resource* b) {
                                  return a->id() != b->id() ? a->id() < b->id() : a < b;
                              });
                    flush_buffer_candidates_.erase(
                        std::unique(flush_buffer_candidates_.begin(), flush_buffer_candidates_.end()),
                        flush_buffer_candidates_.end());
                    for (Resource* r : flush_buffer_candidates_) {
                        visit_buffer(r);
                    }
                    skipped += static_cast<uint64_t>(tracked_.size() - flush_buffer_candidates_.size());
                } else {
                    for (Resource* r : tracked_) {
                        visit_buffer(r);
                    }
                }
                count_work(Work::FlushBuffersVisited, visited);
                count_work(Work::FlushBuffersSkipped, skipped);
                count_work(Work::FlushBuffersUntracked, untracked);
            } else {
                for (Resource* r : tracked_) {
                    r->set_slot_indexed(false);
                    r->flush_shadow(encoder, &mark_pass_);
                }
            }
        } else {
            flush_visit_keys_.clear();
            slot_index_.collect(flush_dirty_slots_, &flush_visit_keys_);
            for (Resource* r : tracked_) {
                if (r->slot_indexed()) {
                    continue;
                }
                r->flush_shadow(encoder, &mark_pass_);
                if (r->window_marks_in_use()) {
                    slot_index_.add(r, r->cpu_address(), r->shadow_bytes());
                    r->set_slot_indexed(true);
                }
            }
            for (const void* p : flush_visit_keys_) {
                Resource* r = const_cast<Resource*>(static_cast<const Resource*>(p));
                r->flush_shadow(encoder, &mark_pass_);
                if (!r->window_marks_in_use()) {
                    slot_index_.remove(r);
                    r->set_slot_indexed(false);
                }
            }
        }
    }
    if (indexed && !crosscheck) {
        count_work(Work::FlushBufferNs, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                std::chrono::steady_clock::now() - buffers_t0)
                                                                .count()));
    }
    // G68 (#663): propagate the same submission MarkPass's dirty slots to tracked window textures
    // so textures sharing a 64 KiB slot with buffers (or written during a submission that did not
    // bind them) never lose write marks.
    {
        std::lock_guard<std::mutex> lock(tracked_textures_mutex_);
        if (!scanned) {
            if (texture_slot_index_.size() != 0) {
                texture_slot_index_.clear();
                for (auto& [id, tex] : tracked_textures_) {
                    tex.window_indexed = false;
                    tex.window_checked = false;
                    tex.window_dirty = true;
                }
            }
        } else if (!flush_dirty_slots_.empty() && texture_slot_index_.size() != 0) {
            flush_visit_keys_.clear();
            texture_slot_index_.collect(flush_dirty_slots_, &flush_visit_keys_);
            for (const void* key : flush_visit_keys_) {
                const uint32_t tex_id = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(key));
                if (auto it = tracked_textures_.find(tex_id); it != tracked_textures_.end()) {
                    it->second.window_dirty = true;
                }
            }
        }
    }
    // Relocated uniform ranges: refresh the aligned copies whose guest bytes changed.
    const auto uniforms_t0 = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(relocation_mutex_);
        flush_visit_keys_.clear();
        if (indexed) {
            uniform_page_index_.collect_pages(flush_dirty_pages_, os::kWrittenPageSize, &flush_visit_keys_);
            for (const auto& h : flush_taken_failed_) {
                uniform_page_index_.collect_range(h.first, h.second, &flush_visit_keys_);
            }
            for (const void* key : flush_visit_keys_) {
                const_cast<RelocatedUniform*>(static_cast<const RelocatedUniform*>(key))->visit_seq = seq;
            }
            for (const void* key : crosscheck_uniforms_) {
                const RelocatedUniform* r = static_cast<const RelocatedUniform*>(key);
                if (r->visit_seq == seq || !r->primed) {
                    count_work(Work::FlushCrosscheckLate);
                } else {
                    count_work(Work::FlushCrosscheckMisses);
                    std::fprintf(stderr, "[flush-crosscheck] uniform range 0x%llx (%llu bytes) changed unreported\n",
                                 static_cast<unsigned long long>(r->va), static_cast<unsigned long long>(r->size));
                }
            }
            crosscheck_uniforms_.clear();
        }
        uint64_t compared = 0, skipped = 0;
        auto visit_uniform = [&](RelocatedUniform& r) {
            // (Covered again this time: the buffer around it may have left the index since.)
            uintptr_t u_lo = 0, u_hi = 0;
            const bool skip = indexed && r.primed && r.visit_seq != seq && uniform_page_index_.range_of(&r, &u_lo, &u_hi) &&
                              ranges_cover(flush_taken_ok_, u_lo, u_hi);
            if (skip && !crosscheck) {
                ++skipped;
                return;
            }
            if (crosscheck && !skip && r.primed && r.visit_seq != seq) {
                // The indexed flush below visits only the reported and the unprimed entries: this one
                // needed a comparison although it is neither.
                count_work(Work::FlushCrosscheckMisses);
                std::fprintf(stderr, "[flush-crosscheck] uniform range 0x%llx (%llu bytes) not a flush candidate\n",
                             static_cast<unsigned long long>(r.va), static_cast<unsigned long long>(r.size));
            }
            const uint64_t cpu = cpu_address_for_gpu(r.va);
            if (!indexed && unwritten(cpu, r.size) && !r.shadow.empty()) {
                return;
            }
            ++compared;
            if (skip) {
                ++skipped;
            }
            // R40: compare the live bytes in place where they are mapped in one piece (no copy
            // through the arena's checked read for an unchanged range, the usual case).
            // (The view holds the arena lock: another guest thread must not decommit the pages mid-read.)
            bool unchanged = false;
            bool have_bytes = false;
            uintptr_t host = 0;
            {
                const com::GuestRangeView view(cpu, static_cast<size_t>(r.size));
                if (view.data() != nullptr) {
                    host = reinterpret_cast<uintptr_t>(view.data());
                    if (r.shadow.size() == r.size && std::memcmp(view.data(), r.shadow.data(), r.shadow.size()) == 0) {
                        unchanged = true;
                    } else {
                        flush_bytes_scratch_.resize(static_cast<size_t>(r.size));  // only the view's bytes: the home past them stays zero
                        std::memcpy(flush_bytes_scratch_.data(), view.data(), flush_bytes_scratch_.size());
                        have_bytes = true;
                    }
                }
            }
            if (!unchanged && !have_bytes) {
                flush_bytes_scratch_.resize(static_cast<size_t>(r.size));
                have_bytes = com::read_guest(cpu, flush_bytes_scratch_.data(), flush_bytes_scratch_.size());
            }
            bool changed = false;
            if (!unchanged && have_bytes && flush_bytes_scratch_ != r.shadow) {
                encoder.write_buffer(r.buffer_id, static_cast<uint32_t>(r.offset), flush_bytes_scratch_.data(),
                                     static_cast<uint32_t>(r.size));
                r.shadow.assign(flush_bytes_scratch_.begin(), flush_bytes_scratch_.end());
                note_upload(r.size);
                changed = true;
            }
            if (!indexed) {
                return;
            }
            if (skip && changed) {
                crosscheck_uniforms_.insert(&r);
            }
            // Indexed by its host range once it has been compared in place (a range read only through
            // the checked copy, or never written, is compared every time).
            if (host != 0 && !r.shadow.empty()) {
                const uintptr_t hi = host + static_cast<uintptr_t>(r.size);
                uniform_page_index_.add(&r, host, hi);
                r.primed = ranges_cover(flush_taken_ok_, host, hi);
            } else {
                uniform_page_index_.remove(&r);
                r.primed = false;
            }
            if (r.primed) {
                uniform_unprimed_.erase(&r);
            } else {
                uniform_unprimed_.insert(&r);
            }
        };
        if (indexed && !crosscheck) {
            // Only the entries a take reported (visit_seq == seq) and the unprimed ones can need a
            // comparison: a primed entry is indexed, so its range lies in a take range that either
            // succeeded (and named none of its pages) or failed (collect_range above visited it).
            // Visiting just those makes this loop O(reported) instead of O(relocations_); the
            // --flush-crosscheck flushes walk every entry and count a needed one that is neither.
            flush_uniform_candidates_.clear();
            flush_uniform_candidates_.reserve(flush_visit_keys_.size() + uniform_unprimed_.size());
            for (const void* key : flush_visit_keys_) {
                flush_uniform_candidates_.push_back(const_cast<RelocatedUniform*>(static_cast<const RelocatedUniform*>(key)));
            }
            flush_uniform_candidates_.insert(flush_uniform_candidates_.end(), uniform_unprimed_.begin(), uniform_unprimed_.end());
            // Deterministic order (guest VA, then size); each entry has its own home, so the uploads'
            // order does not matter for the result.
            std::sort(flush_uniform_candidates_.begin(), flush_uniform_candidates_.end(),
                      [](const RelocatedUniform* a, const RelocatedUniform* b) {
                          return a->va != b->va ? a->va < b->va : (a->size != b->size ? a->size < b->size : a < b);
                      });
            flush_uniform_candidates_.erase(
                std::unique(flush_uniform_candidates_.begin(), flush_uniform_candidates_.end()),
                flush_uniform_candidates_.end());
            for (RelocatedUniform* r : flush_uniform_candidates_) {
                visit_uniform(*r);
            }
            skipped += static_cast<uint64_t>(relocations_.size() - flush_uniform_candidates_.size());
        } else {
            for (auto& [key, r] : relocations_) {
                visit_uniform(r);
            }
        }
        if (indexed) {
            count_work(Work::FlushUniformsCompared, compared);
            count_work(Work::FlushUniformsSkipped, skipped);
        }
    }
    if (indexed && !crosscheck) {
        count_work(Work::FlushUniformNs, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                 std::chrono::steady_clock::now() - uniforms_t0)
                                                                 .count()));
    }
    // Relocated storage buffers (G51): refresh any relocated range whose guest bytes changed.
    {
        std::unique_lock<std::mutex> lock(storage_relocation_mutex_);
        flush_visit_keys_.clear();
        flush_storage_job_count_ = 0;
        // Indexed flushes hash the ranges that may have changed without the lock (StorageHashJob).
        const bool defer_hashes = indexed && !crosscheck;
        if (indexed) {
            // As the uniform ranges above: the entries a take reported (or whose take failed) are
            // visited; a primed entry no take named is unchanged since its last comparison.
            storage_page_index_.collect_pages(flush_dirty_pages_, os::kWrittenPageSize, &flush_visit_keys_);
            for (const auto& h : flush_taken_failed_) {
                storage_page_index_.collect_range(h.first, h.second, &flush_visit_keys_);
            }
            for (const void* key : flush_visit_keys_) {
                const_cast<RelocatedStorage*>(static_cast<const RelocatedStorage*>(key))->visit_seq = seq;
            }
            for (const void* key : crosscheck_storage_) {
                const RelocatedStorage* r = static_cast<const RelocatedStorage*>(key);
                if (r->visit_seq == seq || !r->index_primed) {
                    count_work(Work::FlushCrosscheckLate);
                } else {
                    count_work(Work::FlushCrosscheckMisses);
                    std::fprintf(stderr, "[flush-crosscheck] storage range 0x%llx (%llu bytes) changed unreported\n",
                                 static_cast<unsigned long long>(r->va), static_cast<unsigned long long>(r->size));
                }
            }
            crosscheck_storage_.clear();
        }
        uint64_t storage_compared = 0, storage_skipped = 0;
        // Indexed flushes: (re-)index the range at the host address it was looked at (primed when this
        // flush's take covered it), or drop it from the index when it is not mapped in one piece.
        const auto reindex_storage = [&](RelocatedStorage& r, uintptr_t seen_host, bool index_skip, bool uploaded) {
            if (!indexed) {
                return;
            }
            if (index_skip && uploaded) {
                crosscheck_storage_.insert(&r);
            }
            if (seen_host != 0) {
                const uintptr_t hi = seen_host + static_cast<uintptr_t>(r.size);
                storage_page_index_.add(&r, seen_host, hi);
                r.index_primed = r.ever_uploaded && ranges_cover(flush_taken_ok_, seen_host, hi);
            } else {
                storage_page_index_.remove(&r);
                r.index_primed = false;
            }
            if (r.index_primed) {
                storage_unprimed_.erase(&r);
            } else {
                storage_unprimed_.insert(&r);
            }
        };
        auto visit_storage = [&](RelocatedStorage& r) {
            bool index_skip = false;
            if (indexed && r.index_primed && r.ever_uploaded && r.visit_seq != seq) {
                uintptr_t lo = 0, hi = 0;
                index_skip = storage_page_index_.range_of(&r, &lo, &hi) && ranges_cover(flush_taken_ok_, lo, hi);
            }
            if (indexed) {
                ++(index_skip ? storage_skipped : storage_compared);
            }
            if (index_skip && !crosscheck) {
                return;
            }
            uintptr_t seen_host = 0;
            bool uploaded = false;
            const auto reindex = [&]() { reindex_storage(r, seen_host, index_skip, uploaded); };
            const uint64_t write_len = align_up(r.size, 4u);
            const uint64_t cpu = cpu_address_for_gpu(r.va);
            const bool marks_vouch = r.ever_uploaded && !r.hash_next_flush;  // hashed below otherwise
            r.hash_next_flush = false;
            if (unwritten(cpu, r.size) && marks_vouch) {
                reindex();
                return;
            }
            // R40: hash the live bytes in place where they are mapped in one piece and copy them
            // only when they changed (most ranges are unchanged at most submissions).
            uint64_t hash = 0;
            bool mapped = false;
            bool unchanged = false;
            bool defer = false;
            {
                const com::GuestRangeView view(cpu, static_cast<size_t>(r.size));  // holds the arena lock
                if (view.data() != nullptr) {
                    mapped = true;
                    seen_host = reinterpret_cast<uintptr_t>(view.data());
                    if (native_unwritten(r.write_cursor, r.write_primed, view.data(), r.size) && marks_vouch) {
                        unchanged = true;  // no page written since the last hash: same bytes
                    } else if (defer_hashes) {
                        defer = true;
                    } else {
                        hash = content_hash(view.data(), static_cast<size_t>(r.size));
                        if (r.ever_uploaded && hash == r.last_hash) {
                            unchanged = true;
                        } else {
                            flush_bytes_scratch_.assign(view.data(), view.data() + r.size);
                        }
                    }
                }
            }
            if (unchanged) {
                reindex();
                return;
            }
            if (defer || (!mapped && defer_hashes)) {
                if (flush_storage_job_count_ == flush_storage_jobs_.size()) {
                    flush_storage_jobs_.emplace_back();
                }
                StorageHashJob& job = flush_storage_jobs_[flush_storage_job_count_++];
                job.entry = &r;
                job.va = r.va;
                job.size = r.size;
                job.dedicated = r.dedicated;
                job.created_epoch = r.created_epoch;
                job.cpu = cpu;
                job.taken_host = seen_host;
                job.ever_uploaded = r.ever_uploaded;
                job.last_hash = r.last_hash;
                job.buffer_id = r.buffer_id;
                job.offset = r.offset;
                job.packets_begin = job.packets_end = 0;
                job.upload = false;
                return;
            }
            if (!mapped) {
                flush_bytes_scratch_.resize(static_cast<size_t>(r.size));
                if (!read_guest_best_effort(cpu, flush_bytes_scratch_.data(), flush_bytes_scratch_.size())) {
                    reindex();
                    return;
                }
                hash = content_hash(flush_bytes_scratch_.data(), flush_bytes_scratch_.size());
                if (r.ever_uploaded && hash == r.last_hash) {
                    reindex();
                    return;
                }
            }
            flush_bytes_scratch_.resize(static_cast<size_t>(write_len), 0);
            for (uint64_t off = 0; off < write_len; off += kMaxWriteChunk) {
                const uint64_t n = std::min<uint64_t>(kMaxWriteChunk, write_len - off);
                encoder.write_buffer(r.buffer_id, static_cast<uint32_t>(r.offset + off),
                                     flush_bytes_scratch_.data() + off, static_cast<uint32_t>(n));
            }
            count_work(Work::BufferUploads);
            note_upload(write_len);
            r.last_hash = hash;
            r.ever_uploaded = true;
            uploaded = true;
            reindex();
        };
        if (indexed && !crosscheck) {
            // Only the entries a take reported (visit_seq == seq) and the unprimed ones can need a
            // comparison (as the uniform ranges): O(reported) instead of O(storage_relocations_).
            flush_storage_candidates_.clear();
            flush_storage_candidates_.reserve(flush_visit_keys_.size() + storage_unprimed_.size());
            for (const void* key : flush_visit_keys_) {
                flush_storage_candidates_.push_back(const_cast<RelocatedStorage*>(static_cast<const RelocatedStorage*>(key)));
            }
            flush_storage_candidates_.insert(flush_storage_candidates_.end(), storage_unprimed_.begin(), storage_unprimed_.end());
            // Deterministic order (guest VA, then size); each entry has its own home.
            std::sort(flush_storage_candidates_.begin(), flush_storage_candidates_.end(),
                      [](const RelocatedStorage* a, const RelocatedStorage* b) {
                          return a->va != b->va ? a->va < b->va : (a->size != b->size ? a->size < b->size : a < b);
                      });
            flush_storage_candidates_.erase(
                std::unique(flush_storage_candidates_.begin(), flush_storage_candidates_.end()),
                flush_storage_candidates_.end());
            for (RelocatedStorage* r : flush_storage_candidates_) {
                visit_storage(*r);
            }
            storage_skipped += static_cast<uint64_t>(storage_relocations_.size() - flush_storage_candidates_.size());
            if (flush_storage_job_count_ != 0) {
                // The hashes (and copies of changed ranges) run without the lock, so recording threads
                // binding storage ranges and relocation misses are not held behind them. The write marks
                // and cursors were taken above, before the bytes are read, as on the locked path: a write
                // racing the hash is reported to the next flush.
                lock.unlock();
                // A changed range is encoded straight from the bytes it was hashed from (one copy, while
                // they are still in cache) to the home taken under the lock; the packets of an entry found
                // stale when applying are patched to Nops below.
                const auto encode_job = [&](StorageHashJob& job, const uint8_t* src) {
                    const uint64_t write_len = align_up(job.size, 4u);
                    job.packets_begin = encoder.size();
                    for (uint64_t off = 0; off < write_len; off += kMaxWriteChunk) {
                        const uint64_t n = std::min<uint64_t>(kMaxWriteChunk, write_len - off);
                        const uint8_t* chunk = src + off;
                        if (off + n > job.size) {  // the last chunk, padded to 4 with zeros
                            flush_bytes_scratch_.assign(static_cast<size_t>(n), 0);
                            std::memcpy(flush_bytes_scratch_.data(), src + off, static_cast<size_t>(job.size - off));
                            chunk = flush_bytes_scratch_.data();
                        }
                        encoder.write_buffer(job.buffer_id, static_cast<uint32_t>(job.offset + off), chunk,
                                             static_cast<uint32_t>(n));
                    }
                    job.packets_end = encoder.size();
                };
                const auto hash_job = [&](StorageHashJob& job, const uint8_t* host) {
                    job.host = reinterpret_cast<uintptr_t>(host);
                    job.read_ok = true;
                    job.hash = content_hash(host, static_cast<size_t>(job.size));
                    job.changed = !job.ever_uploaded || job.hash != job.last_hash;
                    if (job.changed) {
                        encode_job(job, host);
                    }
                };
                // The jobs are in guest VA order: neighbouring ranges share one view (one arena lock round
                // trip) when their span is mapped in one piece, bounded so the arena lock is not held long.
                constexpr uint64_t kRunGap = uint64_t{64} << 10;
                constexpr uint64_t kRunMax = uint64_t{4} << 20;
                for (size_t i = 0; i < flush_storage_job_count_;) {
                    const uint64_t lo = flush_storage_jobs_[i].cpu;
                    uint64_t hi = lo + flush_storage_jobs_[i].size;
                    size_t end = i + 1;
                    while (end < flush_storage_job_count_) {
                        const StorageHashJob& next = flush_storage_jobs_[end];
                        if (next.cpu < lo || next.cpu > hi + kRunGap) {
                            break;
                        }
                        const uint64_t next_hi = std::max(hi, next.cpu + next.size);
                        if (next_hi - lo > kRunMax) {
                            break;
                        }
                        hi = next_hi;
                        ++end;
                    }
                    for (size_t k = i; k < end; ++k) {
                        StorageHashJob& job = flush_storage_jobs_[k];
                        job.host = 0;
                        job.read_ok = false;
                        job.changed = false;
                    }
                    bool done = false;
                    if (end - i > 1 && lo != 0) {
                        const com::GuestRangeView view(lo, static_cast<size_t>(hi - lo));
                        if (view.data() != nullptr) {
                            for (size_t k = i; k < end; ++k) {
                                StorageHashJob& job = flush_storage_jobs_[k];
                                hash_job(job, view.data() + (job.cpu - lo));
                            }
                            done = true;
                        }
                    }
                    for (size_t k = i; k < end && !done; ++k) {
                        StorageHashJob& job = flush_storage_jobs_[k];
                        {
                            const com::GuestRangeView view(job.cpu, static_cast<size_t>(job.size));
                            if (view.data() != nullptr) {
                                hash_job(job, view.data());
                            }
                        }
                        if (job.host == 0) {
                            job.bytes.resize(static_cast<size_t>(job.size));
                            job.read_ok = read_guest_best_effort(job.cpu, job.bytes.data(), job.bytes.size());
                            if (job.read_ok) {
                                job.hash = content_hash(job.bytes.data(), job.bytes.size());
                                job.changed = !job.ever_uploaded || job.hash != job.last_hash;
                                if (job.changed) {
                                    encode_job(job, job.bytes.data());
                                }
                            }
                        }
                    }
                    i = end;
                }
                lock.lock();
                for (size_t i = 0; i < flush_storage_job_count_; ++i) {
                    StorageHashJob& job = flush_storage_jobs_[i];
                    // Evicted (or evicted and placed again) meanwhile: the new entry is unprimed.
                    const auto it = storage_relocations_.find(RelocatedStorageKey{job.va, job.size, job.dedicated});
                    if (it == storage_relocations_.end() || &it->second != job.entry ||
                        it->second.created_epoch != job.created_epoch) {
                        // Its home may belong to another range by now: drop the packets encoded for it.
                        for (size_t at = job.packets_begin; at < job.packets_end;) {
                            GpuPacketHeader header{};
                            std::memcpy(&header, encoder.data() + at, sizeof(header));
                            header.opcode = static_cast<uint16_t>(GpuOpcode::Nop);
                            encoder.patch(at, &header, sizeof(header));
                            at += sizeof(header) + header.payload_bytes;
                        }
                        continue;
                    }
                    RelocatedStorage& r = it->second;
                    if (job.host != job.taken_host) {
                        r.write_primed = false;  // remapped since the cursor was taken: take it again
                    }
                    if (job.changed) {
                        r.last_hash = job.hash;
                        r.ever_uploaded = true;
                        job.upload = true;
                    }
                    reindex_storage(r, job.read_ok ? job.host : 0, false, job.upload);
                }
            }
        } else {
            for (auto& [key, r] : storage_relocations_) {
                visit_storage(r);
            }
        }
        if (indexed) {
            count_work(Work::FlushStorageCompared, storage_compared);
            count_work(Work::FlushStorageSkipped, storage_skipped);
        }
    }
    // The changed ranges hashed without the lock (their packets are encoded already). A home handed to
    // another range after this point is the same case as on the locked path (whose packets were encoded
    // but not yet submitted either).
    for (size_t i = 0; i < flush_storage_job_count_; ++i) {
        StorageHashJob& job = flush_storage_jobs_[i];
        if (job.upload) {
            count_work(Work::BufferUploads);
            note_upload(align_up(job.size, 4u));
        }
        if (job.bytes.capacity() > (size_t{1} << 20)) {
            std::vector<uint8_t>().swap(job.bytes);  // the slots are reused: do not keep large copies
        }
    }
    // Relocated index / vertex buffers (G58): re-upload ring buffers the title wrote since.
    {
        std::lock_guard<std::mutex> lock(ia_relocation_mutex_);
        flush_visit_keys_.clear();
        if (indexed) {
            // As the uniform and storage ranges above: the entries a take reported (or whose take
            // failed) are visited; a primed entry no take named is unchanged since its last comparison.
            ia_page_index_.collect_pages(flush_dirty_pages_, os::kWrittenPageSize, &flush_visit_keys_);
            for (const auto& h : flush_taken_failed_) {
                ia_page_index_.collect_range(h.first, h.second, &flush_visit_keys_);
            }
            for (const void* key : flush_visit_keys_) {
                const_cast<RelocatedIa*>(static_cast<const RelocatedIa*>(key))->visit_seq = seq;
            }
            for (const void* key : crosscheck_ia_) {
                const RelocatedIa* r = static_cast<const RelocatedIa*>(key);
                if (r->visit_seq == seq || !r->index_primed) {
                    count_work(Work::FlushCrosscheckLate);
                } else {
                    count_work(Work::FlushCrosscheckMisses);
                    std::fprintf(stderr, "[flush-crosscheck] ia range 0x%llx (%llu bytes) changed unreported\n",
                                 static_cast<unsigned long long>(r->va), static_cast<unsigned long long>(r->size));
                }
            }
            crosscheck_ia_.clear();
        }
        uint64_t ia_compared = 0, ia_skipped = 0;
        auto visit_ia = [&](RelocatedIa& r) {
            bool index_skip = false;
            if (indexed && r.index_primed && r.visit_seq != seq) {
                uintptr_t lo = 0, hi = 0;
                index_skip = ia_page_index_.range_of(&r, &lo, &hi) && ranges_cover(flush_taken_ok_, lo, hi);
            }
            if (indexed) {
                ++(index_skip ? ia_skipped : ia_compared);
            }
            if (index_skip && !crosscheck) {
                return;
            }
            if (crosscheck && !index_skip && r.index_primed && r.visit_seq != seq) {
                count_work(Work::FlushCrosscheckMisses);
                std::fprintf(stderr, "[flush-crosscheck] ia range 0x%llx (%llu bytes) not a flush candidate\n",
                             static_cast<unsigned long long>(r.va), static_cast<unsigned long long>(r.size));
            }
            uintptr_t seen_host = 0;
            bool uploaded = false;
            const auto reindex = [&]() {
                if (!indexed) {
                    return;
                }
                if (index_skip && uploaded) {
                    crosscheck_ia_.insert(&r);
                }
                if (seen_host != 0) {
                    const uintptr_t hi = seen_host + static_cast<uintptr_t>(r.size);
                    ia_page_index_.add(&r, seen_host, hi);
                    r.index_primed = ranges_cover(flush_taken_ok_, seen_host, hi);
                } else {
                    ia_page_index_.remove(&r);
                    r.index_primed = false;
                }
                if (r.index_primed) {
                    ia_unprimed_.erase(&r);
                } else {
                    ia_unprimed_.insert(&r);
                }
            };
            const uint64_t cpu = cpu_address_for_gpu(r.va);
            if (unwritten(cpu, r.size)) {
                reindex();
                return;  // (uploaded when it was created)
            }
            // R40: as above, hash in place and copy only a changed range.
            uint64_t hash = 0;
            bool mapped = false;
            bool unchanged = false;
            {
                const com::GuestRangeView view(cpu, static_cast<size_t>(r.size));  // holds the arena lock
                if (view.data() != nullptr) {
                    mapped = true;
                    seen_host = reinterpret_cast<uintptr_t>(view.data());
                    if (native_unwritten(r.write_cursor, r.write_primed, view.data(), r.size)) {
                        unchanged = true;  // no page written since the last hash: same bytes
                    } else {
                        hash = content_hash(view.data(), static_cast<size_t>(r.size));
                        if (hash == r.last_hash) {
                            unchanged = true;
                        } else {
                            flush_bytes_scratch_.assign(view.data(), view.data() + r.size);
                        }
                    }
                }
            }
            if (unchanged) {
                reindex();
                return;
            }
            if (!mapped) {
                flush_bytes_scratch_.resize(static_cast<size_t>(r.size));
                if (!read_guest_best_effort(cpu, flush_bytes_scratch_.data(), flush_bytes_scratch_.size())) {
                    reindex();
                    return;
                }
                hash = content_hash(flush_bytes_scratch_.data(), flush_bytes_scratch_.size());
                if (hash == r.last_hash) {
                    reindex();
                    return;
                }
            }
            flush_bytes_scratch_.resize(static_cast<size_t>(r.buffer_bytes), 0);
            for (uint64_t off = 0; off < r.buffer_bytes; off += kMaxWriteChunk) {
                const uint64_t n = std::min<uint64_t>(kMaxWriteChunk, r.buffer_bytes - off);
                encoder.write_buffer(r.buffer_id, static_cast<uint32_t>(off), flush_bytes_scratch_.data() + off,
                                     static_cast<uint32_t>(n));
            }
            count_work(Work::BufferUploads);
            note_upload(r.buffer_bytes);
            r.last_hash = hash;
            uploaded = true;
            reindex();
        };
        if (indexed && !crosscheck) {
            flush_ia_candidates_.clear();
            flush_ia_candidates_.reserve(flush_visit_keys_.size() + ia_unprimed_.size());
            for (const void* key : flush_visit_keys_) {
                flush_ia_candidates_.push_back(const_cast<RelocatedIa*>(static_cast<const RelocatedIa*>(key)));
            }
            flush_ia_candidates_.insert(flush_ia_candidates_.end(), ia_unprimed_.begin(), ia_unprimed_.end());
            std::sort(flush_ia_candidates_.begin(), flush_ia_candidates_.end(),
                      [](const RelocatedIa* a, const RelocatedIa* b) {
                          return a->va != b->va ? a->va < b->va : (a->size != b->size ? a->size < b->size : a < b);
                      });
            flush_ia_candidates_.erase(
                std::unique(flush_ia_candidates_.begin(), flush_ia_candidates_.end()),
                flush_ia_candidates_.end());
            for (RelocatedIa* r : flush_ia_candidates_) {
                visit_ia(*r);
            }
            ia_skipped += static_cast<uint64_t>(ia_relocations_.size() - flush_ia_candidates_.size());
        } else {
            for (auto& [key, r] : ia_relocations_) {
                visit_ia(r);
            }
        }
        if (indexed) {
            count_work(Work::FlushIaCompared, ia_compared);
            count_work(Work::FlushIaSkipped, ia_skipped);
        }
    }
    if (flush_bytes_scratch_.capacity() > (uint64_t{1u} << 20)) {
        std::vector<uint8_t>().swap(flush_bytes_scratch_);
    }
}

bool Device::relocate_uniform(uint64_t va, uint64_t size, uint64_t binding_size, uint32_t* buffer_id,
                              uint64_t* offset) {
    constexpr uint64_t kChunk = 1u << 20;
    const uint64_t need = align_up(std::max(size, binding_size), bridge::kUniformOffsetAlignment);
    if (size == 0 || need > (64u << 10)) {
        return false;
    }
    uint8_t probe = 0;
    if (!com::read_guest(cpu_address_for_gpu(va), &probe, 1)) {
        return false;  // GPU-only memory: no bytes to copy
    }
    std::lock_guard<std::mutex> lock(relocation_mutex_);
    auto it = relocations_.find({va, size, need});
    if (it == relocations_.end()) {
        constexpr size_t kMaxEntries = 8192;  // O36: per-draw VAs (ring-allocated constants) must not grow this forever
        if (relocations_.size() >= kMaxEntries && !relocation_lru_.empty()) {
            const auto lru = relocations_.find(relocation_lru_.front());
            relocation_lru_.pop_front();
            relocation_free_[{lru->second.size, std::get<2>(lru->first)}].emplace_back(lru->second.buffer_id,
                                                                                         lru->second.offset);
            uniform_page_index_.remove(&lru->second);
            crosscheck_uniforms_.erase(&lru->second);
            uniform_unprimed_.erase(&lru->second);
            relocations_.erase(lru);
            relocation_evictions_.fetch_add(1, std::memory_order_relaxed);
        }
        RelocatedUniform r;
        r.va = va;
        r.size = size;
        auto& free_homes = relocation_free_[{size, need}];
        if (!free_homes.empty()) {
            r.buffer_id = free_homes.back().first;
            r.offset = free_homes.back().second;
            free_homes.pop_back();
        } else {
            // A fresh home: the chunk is zero-initialised, so the bytes past `size` stay zero
            // (zero-extension for WebGPU's minimum binding size, not data).
            if (relocation_buffer_ == 0 || relocation_used_ + need > kChunk) {
                relocation_buffer_ = allocate_id();
                CommandEncoder enc;
                enc.create_buffer(relocation_buffer_, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst, kChunk);
                submit(enc);
                relocation_used_ = 0;
            }
            r.buffer_id = relocation_buffer_;
            r.offset = relocation_used_;
            relocation_used_ += need;
        }
        it = relocations_.emplace(std::make_tuple(va, size, need), std::move(r)).first;
        it->second.lru_pos = relocation_lru_.insert(relocation_lru_.end(), it->first);
        uniform_unprimed_.insert(&it->second);  // new: compared at the next flush
    } else {
        relocation_lru_.splice(relocation_lru_.end(), relocation_lru_, it->second.lru_pos);  // most recent
    }
    *buffer_id = it->second.buffer_id;
    *offset = it->second.offset;
    return true;
}

size_t Device::address_texture_count() const {
    std::lock_guard<std::mutex> lock(address_textures_mutex_);
    return address_textures_.size();
}

size_t Device::relocated_uniform_count() const {
    std::lock_guard<std::mutex> lock(relocation_mutex_);
    return relocations_.size();
}

bool Device::relocate_storage_buffer(uint64_t va, uint64_t size, uint32_t* buffer_id, uint64_t* offset,
                                     bool require_zero_offset) {
    constexpr uint64_t kStorageOffsetAlignment = 256;
    constexpr uint64_t kChunk = uint64_t{4u} << 20;
    constexpr uint64_t kMaxBytes = uint64_t{16u} << 20;
    constexpr size_t kMaxEntries = kStorageRelocationSoftEntries;
    // A home is only handed to another range once the presents since its last use guarantee that
    // every submission that bound it is behind a Present in the packet stream (#663): the browser
    // worker applies a frame's buffer writes before that frame's draws, and a title recording one
    // frame ahead binds a home one present before the submission that reads it executes.
    constexpr uint64_t kReuseAfterPresents = 3;
    // Past this many entries (or bytes of homes) the least recently used one is evicted even if it
    // is recent.
    constexpr size_t kHardMaxEntries = kStorageRelocationHardEntries;
    constexpr uint64_t kHardMaxBytes = kStorageRelocationHardBytes;

    if (va == 0 || size == 0 || size > kMaxBytes || buffer_id == nullptr || offset == nullptr) {
        return false;
    }
    const uint64_t cpu_va = cpu_address_for_gpu(va);
    uint8_t probe = 0;
    if (!com::read_guest(cpu_va, &probe, 1)) {
        return false;  // unmapped / GPU-only memory: no bytes to copy
    }

    const uint64_t need = align_up(size, kStorageOffsetAlignment);
    const bool dedicated = require_zero_offset || need > kChunk;
    const RelocatedStorageKey key{va, size, dedicated};

    const uint64_t present = present_epoch();
    // A hit only touches the LRU; false on a miss. Called with storage_relocation_mutex_ held.
    const auto take_hit = [&]() {
        const auto it = storage_relocations_.find(key);
        if (it == storage_relocations_.end()) {
            return false;
        }
        touch_storage_lru(it->second, present);
        *buffer_id = it->second.buffer_id;
        *offset = it->second.offset;
        return true;
    };
    {
        std::lock_guard<std::mutex> lock(storage_relocation_mutex_);
        if (take_hit()) {
            return true;
        }
    }

    // Miss: read and hash the guest bytes (up to 16 MiB) without the lock, so recording threads that
    // hit (and refresh_memory_bind_groups) are not held behind the copy. A guest write racing the read
    // is caught like any other: the new entry is unprimed and compared at the next flush. If a flush ran
    // during the read it may have taken the write marks of a write that landed after it, so the entry is
    // then marked hash_next_flush.
    const uint64_t flush_epoch_before = flush_epoch_.load(std::memory_order_acquire);
    std::vector<uint8_t> bytes(static_cast<size_t>(size), 0);
    if (!read_guest_best_effort(cpu_va, bytes.data(), bytes.size())) {
        return false;
    }
    const uint64_t hash = content_hash(bytes.data(), bytes.size());
    const uint64_t write_len = align_up(size, 4u);
    bytes.resize(static_cast<size_t>(write_len), 0);
    // The upload is encoded here too (home id / offset patched in once the home is chosen below), so
    // the lock is not held over the copy of the bytes into packets.
    CommandEncoder writes;
    std::vector<size_t> write_packets;  // offsets of the WriteBuffer packets in `writes`
    for (uint64_t off = 0; off < write_len; off += kMaxWriteChunk) {
        const uint64_t n = std::min<uint64_t>(kMaxWriteChunk, write_len - off);
        write_packets.push_back(writes.size());
        writes.write_buffer(0, 0, bytes.data() + off, static_cast<uint32_t>(n));
    }

    std::unique_lock<std::mutex> lock(storage_relocation_mutex_);
    if (take_hit()) {
        return true;  // another thread placed the same range meanwhile
    }
    count_work(Work::StorageRelocationMisses);

    CommandEncoder enc;
    uint32_t target_buf = 0;
    uint64_t target_off = 0;
    uint64_t target_cap = 0;

    std::vector<uint32_t> emptied_chunks;  // shared chunks that lost a home in this call (#2093)
    CommandEncoder released;               // their destroys
    const auto erase_entry = [&](std::map<RelocatedStorageKey, RelocatedStorage>::iterator it) {
        storage_relocation_lru_.erase(&it->second.lru_node);
        storage_relocation_bytes_ -= std::min(storage_relocation_bytes_, it->second.slot_capacity);
        if (!it->second.dedicated) {
            const uint32_t chunk = it->second.buffer_id;
            auto live_it = storage_relocation_chunk_live_.find(chunk);
            if (live_it != storage_relocation_chunk_live_.end() && live_it->second > 0) {
                --live_it->second;
            }
            emptied_chunks.push_back(chunk);
        }
        storage_home_index_.erase(storage_home_key(it->second.buffer_id, it->second.offset));
        storage_page_index_.remove(&it->second);
        crosscheck_storage_.erase(&it->second);
        storage_unprimed_.erase(&it->second);
        storage_relocations_.erase(it);
        relocation_evictions_.fetch_add(1, std::memory_order_relaxed);
    };
    const auto over_hard_bound = [&] {
        return storage_relocations_.size() >= kHardMaxEntries || storage_relocation_bytes_ + need > kHardMaxBytes;
    };
    if (storage_relocations_.size() >= kMaxEntries || over_hard_bound()) {
        const bool forced = over_hard_bound();
        const auto settled = [&](const StorageLru::Node& n) {
            return forced || present >= n.last_used_present + kReuseAfterPresents;
        };
        const auto entry_of = [&](const StorageLru::Node& n) {
            return storage_relocations_.find(RelocatedStorageKey{n.entry->va, n.entry->size, n.entry->dedicated});
        };
        // Oldest first, up to the first home still in use or kStorageReuseScan homes: the first settled
        // home large enough is handed to the new range (indexed, gpu/storage_reuse_lru.h).
        auto reuse_it = storage_relocations_.end();
        size_t looked = 0;
        if (const StorageLru::Node* n =
                storage_relocation_lru_.find_reuse(present, forced, need, dedicated, kReuseAfterPresents, &looked)) {
            reuse_it = entry_of(*n);
        }
        count_work(Work::StorageReuseScanSteps, looked);
        bool dropped = false;
        if (reuse_it != storage_relocations_.end()) {
            target_buf = reuse_it->second.buffer_id;
            target_off = reuse_it->second.offset;
            target_cap = reuse_it->second.slot_capacity;
            erase_entry(reuse_it);
            dropped = true;
        }
        // Past a hard bound the least recently used homes go until the new one fits; past the soft
        // bound one settled home goes. A home a recent submission may still bind is kept otherwise
        // (grow instead of evicting).
        while (!storage_relocation_lru_.empty()) {
            const StorageLru::Node& front = *storage_relocation_lru_.front();
            if (!over_hard_bound() && (dropped || !settled(front))) {
                break;
            }
            const auto lru_it = entry_of(front);
            if (!lru_it->second.dedicated) {
                storage_relocation_free_.emplace(lru_it->second.slot_capacity,
                                                 std::make_pair(lru_it->second.buffer_id, lru_it->second.offset));
            } else {
                const uint32_t old_id = lru_it->second.buffer_id;
                bridge_->unregister_resource(old_id);
                bridge_->advance_present_frame(presents(), enc);
                bridge_->destroy_or_defer_resource(old_id, enc);
                storage_relocation_buffers_.erase(
                    std::remove(storage_relocation_buffers_.begin(), storage_relocation_buffers_.end(), old_id),
                    storage_relocation_buffers_.end());
                note_storage_home_buffer(old_id, false);
            }
            erase_entry(lru_it);
            dropped = true;
        }
    }

    if (target_buf == 0 && !dedicated) {
        const auto fit = storage_relocation_free_.lower_bound(need);
        if (fit != storage_relocation_free_.end() && fit->first <= need * 2) {
            target_cap = fit->first;
            target_buf = fit->second.first;
            target_off = fit->second.second;
            storage_relocation_free_.erase(fit);
        }
    }
    if (target_buf == 0) {
        if (dedicated) {
            target_buf = allocate_id();
            target_off = 0;
            target_cap = need;
            enc.create_buffer(target_buf, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, target_cap);
            bridge_->register_buffer(target_buf, allocate_gpu_va(target_cap), target_cap);
            storage_relocation_buffers_.push_back(target_buf);
            note_storage_home_buffer(target_buf, true);
        } else {
            if (storage_relocation_buffer_ == 0 || storage_relocation_used_ + need > kChunk) {
                storage_relocation_buffer_ = allocate_id();
                enc.create_buffer(storage_relocation_buffer_, GpuBufferUsage::Storage | GpuBufferUsage::CopyDst, kChunk);
                bridge_->register_buffer(storage_relocation_buffer_, allocate_gpu_va(kChunk), kChunk);
                storage_relocation_buffers_.push_back(storage_relocation_buffer_);
                note_storage_home_buffer(storage_relocation_buffer_, true);
                storage_relocation_used_ = 0;
            }
            target_buf = storage_relocation_buffer_;
            target_off = storage_relocation_used_;
            target_cap = need;
            storage_relocation_used_ += need;
        }
    }

    for (size_t i = 0; i < write_packets.size(); ++i) {
        // id and dst_offset only; data_len was encoded with the bytes.
        const GpuWriteBufferPayload head{target_buf, static_cast<uint32_t>(target_off + i * kMaxWriteChunk), 0u, 0u};
        writes.patch(write_packets[i] + sizeof(GpuPacketHeader), &head, offsetof(GpuWriteBufferPayload, data_len));
    }
    count_work(Work::BufferUploads);
    note_upload(write_len);

    RelocatedStorage entry{};
    entry.va = va;
    entry.size = size;
    entry.buffer_id = target_buf;
    entry.offset = target_off;
    entry.slot_capacity = target_cap;
    entry.last_hash = hash;
    entry.ever_uploaded = true;
    entry.hash_next_flush = (flush_epoch_before & 1u) != 0 ||
                            flush_epoch_.load(std::memory_order_acquire) != flush_epoch_before;
    entry.dedicated = dedicated;
    entry.created_epoch = relocation_evictions_.load(std::memory_order_relaxed);
    const auto placed = storage_relocations_.emplace(key, entry).first;
    storage_home_index_[storage_home_key(target_buf, target_off)] = placed;
    storage_unprimed_.insert(&placed->second);  // new: compared at the next flush
    storage_relocation_lru_.push_back(&placed->second.lru_node, &placed->second, present, target_cap, dedicated);
    storage_relocation_bytes_ += target_cap;
    if (!dedicated) {
        ++storage_relocation_chunk_live_[target_buf];
    }
    // A shared chunk whose homes were all evicted is destroyed (its free slots go with it), so evicted
    // small relocations return their space instead of growing the chunk set over a long session.
    std::sort(emptied_chunks.begin(), emptied_chunks.end());
    emptied_chunks.erase(std::unique(emptied_chunks.begin(), emptied_chunks.end()), emptied_chunks.end());
    for (const uint32_t chunk : emptied_chunks) {
        const auto live_it = storage_relocation_chunk_live_.find(chunk);
        if (live_it == storage_relocation_chunk_live_.end() || live_it->second != 0) {
            continue;
        }
        storage_relocation_chunk_live_.erase(live_it);
        for (auto f = storage_relocation_free_.begin(); f != storage_relocation_free_.end();) {
            f = f->second.first == chunk ? storage_relocation_free_.erase(f) : std::next(f);
        }
        if (chunk == storage_relocation_buffer_) {
            storage_relocation_buffer_ = 0;
            storage_relocation_used_ = 0;
        }
        bridge_->unregister_resource(chunk);
        bridge_->advance_present_frame(presents(), released);
        bridge_->destroy_or_defer_resource(chunk, released);
        storage_relocation_buffers_.erase(
            std::remove(storage_relocation_buffers_.begin(), storage_relocation_buffers_.end(), chunk),
            storage_relocation_buffers_.end());
        note_storage_home_buffer(chunk, false);
    }

    // Not submit(): the sink can block (live pump backpressure). The packets (evicted homes' destroys,
    // the new home's create and upload, emptied chunks' destroys) go out at the start of the next
    // submission, before any command list that binds the home. The queue lock is taken before the
    // relocation lock is let go: a thread that binds the new entry from now on can only submit after
    // these packets are queued, and misses queue in the order they changed the cache. The copy into
    // the queue is then made without holding up the recording threads.
    {
        std::lock_guard<std::mutex> queue(next_submission_mutex_);
        lock.unlock();
        for (const CommandEncoder* e : {&enc, &writes, &released}) {
            next_submission_.insert(next_submission_.end(), e->data(), e->data() + e->size());
        }
    }

    *buffer_id = target_buf;
    *offset = target_off;
    return true;
}

size_t Device::relocated_storage_count() const {
    std::lock_guard<std::mutex> lock(storage_relocation_mutex_);
    return storage_relocations_.size();
}

uint64_t Device::relocated_storage_bytes() const {
    std::lock_guard<std::mutex> lock(storage_relocation_mutex_);
    return storage_relocation_bytes_;
}

size_t Device::relocated_storage_chunk_count() const {
    std::lock_guard<std::mutex> lock(storage_relocation_mutex_);
    size_t dedicated = 0;
    for (const auto& [key, r] : storage_relocations_) {
        dedicated += r.dedicated ? 1 : 0;
    }
    return storage_relocation_buffers_.size() >= dedicated ? storage_relocation_buffers_.size() - dedicated : 0;
}

bool Device::storage_relocation_home(uint32_t buffer_id) const {
    std::lock_guard<std::mutex> lock(storage_home_ids_mutex_);
    return storage_home_ids_.count(buffer_id) != 0;
}

void Device::note_storage_home_buffer(uint32_t buffer_id, bool live) {
    std::lock_guard<std::mutex> lock(storage_home_ids_mutex_);
    if (live) {
        storage_home_ids_.insert(buffer_id);
    } else {
        storage_home_ids_.erase(buffer_id);
    }
}

bool Device::storage_homes_unmoved(const std::pair<uint32_t, uint64_t>* homes, size_t count, uint64_t since_epoch) {
    if (count == 0) {
        return true;
    }
    const uint64_t present = present_epoch();
    std::lock_guard<std::mutex> lock(storage_relocation_mutex_);
    for (size_t i = 0; i < count; ++i) {
        const auto found = storage_home_index_.find(storage_home_key(homes[i].first, homes[i].second));
        if (found == storage_home_index_.end() || found->second->second.created_epoch > since_epoch) {
            return false;  // evicted, or handed to another range since: resolve again
        }
    }
    // Same LRU bookkeeping as a hit in relocate_storage_buffer (the #663 reuse guard counts it).
    for (size_t i = 0; i < count; ++i) {
        touch_storage_lru(storage_home_index_.find(storage_home_key(homes[i].first, homes[i].second))->second->second,
                          present);
    }
    return true;
}

void Device::touch_storage_lru(RelocatedStorage& r, uint64_t present) {
    storage_relocation_lru_.touch(&r.lru_node, present);
}

bool Device::relocate_ia_buffer(uint64_t va, uint64_t size, uint32_t* buffer_id, uint64_t* offset) {
    constexpr uint64_t kMaxBytes = uint64_t{16u} << 20;
    constexpr uint64_t kMaxTotalBytes = uint64_t{256u} << 20;
    constexpr size_t kMaxEntries = 256;
    // WebGPU wants vertex buffer offsets 4-byte aligned and index offsets aligned to the index
    // size (2 or 4), so a sub-range is only served from a covering buffer at a 4-byte aligned offset.
    constexpr uint64_t kOffsetAlignment = 4;

    if (va == 0 || size == 0 || size > kMaxBytes || va + size < va || buffer_id == nullptr || offset == nullptr) {
        return false;
    }
    const uint64_t cpu_va = cpu_address_for_gpu(va);
    uint8_t probe = 0;
    if (!com::read_guest(cpu_va, &probe, 1)) {
        return false;  // unmapped / GPU-only memory: no bytes to copy
    }

    std::lock_guard<std::mutex> lock(ia_relocation_mutex_);
    const uint64_t seq = ++ia_relocation_seq_;
    // A relocation whose [base, base + size) covers the request: the title's ring buffer pointer
    // advanced within a range already mirrored (keys sort by base VA, so scan those <= va).
    for (auto it = ia_relocations_.upper_bound(std::make_pair(va, ~uint64_t{0})); it != ia_relocations_.begin();) {
        --it;
        const uint64_t base = it->first.first;
        const uint64_t mirrored = it->first.second;
        const uint64_t delta = va - base;
        if (delta % kOffsetAlignment == 0 && delta <= mirrored && size <= mirrored - delta) {
            it->second.last_used_seq = seq;
            *buffer_id = it->second.buffer_id;
            *offset = delta;
            return true;
        }
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(size), 0);
    if (!read_guest_best_effort(cpu_va, bytes.data(), bytes.size())) {
        return false;
    }
    const uint64_t hash = content_hash(bytes.data(), bytes.size());
    const uint64_t buffer_bytes = align_up(size, 4u);
    bytes.resize(static_cast<size_t>(buffer_bytes), 0);

    // Bounded like the storage relocations: the least recently bound mirror is destroyed
    // (preferring entries not used in the current frame, and deferring DestroyResource past Present
    // if a current-frame buffer had to be evicted).
    CommandEncoder enc;
    bridge_->advance_present_frame(presents(), enc);
    while (!ia_relocations_.empty() &&
           (ia_relocations_.size() >= kMaxEntries || ia_relocation_bytes_ + buffer_bytes > kMaxTotalBytes)) {
        auto lru = ia_relocations_.end();
        bool lru_in_frame = true;
        uint64_t lru_use = UINT64_MAX;
        {
            const bridge::StateBridgeDevice::ResourceUseScan use_scan(*bridge_);
            for (auto it = ia_relocations_.begin(); it != ia_relocations_.end(); ++it) {
                const bool in_frame = use_scan.in_frame(it->second.buffer_id);
                const uint64_t use = std::max(it->second.last_used_seq, use_scan.last_use_tick(it->second.buffer_id));
                if (lru == ia_relocations_.end() ||
                    (lru_in_frame && !in_frame) ||
                    (in_frame == lru_in_frame && use < lru_use)) {
                    lru = it;
                    lru_in_frame = in_frame;
                    lru_use = use;
                }
            }
        }
        bridge_->destroy_or_defer_resource(lru->second.buffer_id, enc);
        ia_relocation_bytes_ -= lru->second.buffer_bytes;
        ia_page_index_.remove(&lru->second);
        crosscheck_ia_.erase(&lru->second);
        ia_unprimed_.erase(&lru->second);
        ia_relocations_.erase(lru);
    }

    const uint32_t id = allocate_id();
    enc.create_buffer(id, GpuBufferUsage::Vertex | GpuBufferUsage::Index | GpuBufferUsage::CopyDst, buffer_bytes);
    for (uint64_t off = 0; off < buffer_bytes; off += kMaxWriteChunk) {
        const uint64_t n = std::min<uint64_t>(kMaxWriteChunk, buffer_bytes - off);
        enc.write_buffer(id, static_cast<uint32_t>(off), bytes.data() + off, static_cast<uint32_t>(n));
    }
    submit(enc);
    count_work(Work::BufferUploads);
    note_upload(buffer_bytes);

    RelocatedIa entry{};
    entry.va = va;
    entry.size = size;
    entry.buffer_bytes = buffer_bytes;
    entry.buffer_id = id;
    entry.last_hash = hash;
    entry.last_used_seq = seq;
    const auto placed = ia_relocations_.emplace(std::make_pair(va, size), entry).first;
    ia_unprimed_.insert(&placed->second);  // new: compared at the next flush
    ia_relocation_bytes_ += buffer_bytes;

    *buffer_id = id;
    *offset = 0;
    return true;
}

size_t Device::relocated_ia_count() const {
    std::lock_guard<std::mutex> lock(ia_relocation_mutex_);
    return ia_relocations_.size();
}

bool Device::set_back_buffer(uint64_t resource_interface_pointer) {
    com::detail::Resolved r;
    if (!com::detail::lookup_interface(resource_interface_pointer, &r) || r.iface != &kID3D12Resource.info) {
        return false;
    }
    auto* resource = static_cast<Resource*>(static_cast<IResourceHost*>(r.host));
    if (resource->desc().Dimension != ResourceDimension::Texture2D || !resource->has_gpu_texture()) {
        return false;
    }
    resource->mark_back_buffer();
    return true;
}

void Device::note_unsupported_copy(const std::string& message) {
    count_work(Work::CopiesSkipped);
    unsupported_copies_.fetch_add(1, std::memory_order_relaxed);
    note_bad_argument(message);
}

bool Device::upload_needs_conversion(uint32_t dxgi_format, pso::FormatUsage usage) const {
    const pso::TextureFormatMapping m =
        pso::map_texture_format(static_cast<pso::DxgiFormat>(dxgi_format), usage, bridge_->options().features);
    // A8_UNORM is converted on upload (widen_a8_rows).
    return m.supported() && !pso::has_depth_aspect(m.format) && !pso::has_stencil_aspect(m.format) &&
           (m.diagnostics & (pso::FormatDiag::NeedsCpuConversion | pso::FormatDiag::FeatureFallback)) != 0 &&
           static_cast<pso::DxgiFormat>(dxgi_format) != pso::DxgiFormat::A8_UNORM;
}

bool widen_a8_rows(const uint8_t* src, size_t src_size, uint32_t width, uint32_t rows, uint32_t slices,
                   uint32_t src_pitch, uint64_t slice_pitch, std::vector<uint8_t>* out, uint32_t* out_pitch) {
    const uint64_t pitch = align_up(uint64_t{width} * 4u, 256u);
    if (rows == 0 || slices == 0 ||
        uint64_t{slices - 1} * slice_pitch + uint64_t{rows - 1} * src_pitch + width > src_size) {
        return false;
    }
    out->assign(static_cast<size_t>(pitch * rows * slices), 0);
    for (uint32_t z = 0; z < slices; ++z) {
        for (uint32_t r = 0; r < rows; ++r) {
            pso::widen_a8_texels(pso::DxgiFormat::A8_UNORM, pso::WgpuTextureFormat::Rgba8Unorm,
                                 src + z * slice_pitch + uint64_t{r} * src_pitch, width,
                                 out->data() + (uint64_t{z} * rows + r) * pitch);
        }
    }
    *out_pitch = static_cast<uint32_t>(pitch);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Heap
// ---------------------------------------------------------------------------------------------

Heap::Heap(com::Ref<Device> device, const GuestHeapDesc& desc) : ChildBase(std::move(device)), desc_(desc) {
    cpu_visible_ = heap_cpu_visible(desc.Properties);
    const uint64_t size = align_up(std::max<uint64_t>(desc.SizeInBytes, 1), 64u << 10);
    if (cpu_visible_) {
        base_ = com::guest_memory().virtual_alloc(0, size, os::kMemReserve | os::kMemCommit, os::PageProtect::ReadWrite);
        owns_guest_memory_ = base_ != 0;
    } else {
        base_ = device_->allocate_gpu_va(size);
    }
    device_->memory_heaps.fetch_add(1);
}

Heap::~Heap() {
    if (owns_guest_memory_) {
        com::guest_memory().release(base_);
    }
    device_->memory_heaps.fetch_sub(1);
}

// ---------------------------------------------------------------------------------------------
// Resource
// ---------------------------------------------------------------------------------------------

Resource::Resource(com::Ref<Device> device, const GuestResourceDesc& desc, Placement placement, uint32_t initial_state)
    : ChildBase(std::move(device)), desc_(desc), placement_(std::move(placement)), initial_state_(initial_state) {
    device_->resources.fetch_add(1);
}

Resource::~Resource() {
    if (id_ != 0) {
        count_work(Work::ResourcesDestroyed);
    }
    if (probe_registered_) {
        TextureProbe& p = texture_probe();
        std::lock_guard<std::mutex> lock(p.mutex);
        p.textures.erase(std::remove(p.textures.begin(), p.textures.end(), this), p.textures.end());
    }
    if (tracked_) {
        device_->untrack_cpu_buffer(this);
        device_->note_shadow_bytes(-static_cast<int64_t>(shadow_size_), -static_cast<int64_t>(shadow_resident_));
    }
    if (texture_tracked_) {
        device_->untrack_guest_texture(id_);
    }
    if (id_ != 0 && !is_buffer()) {
        texture_budget().unregister_texture(id_);
    }
    if (budget_bytes_ != 0) {
        texture_budget().note_destroyed(budget_bytes_);
        budget_bytes_ = 0;
    }
    if (id_ != 0) {
        device_->state_bridge().unregister_resource(id_);
        device_->resource_states().unregister_resource(id_);
        if (is_buffer() || gpu_texture_) {
            CommandEncoder enc;
            device_->state_bridge().advance_present_frame(device_->presents(), enc);
            // Keep a still-deferred create if any unexecuted command list or in-frame command
            // already references this id: emit_locked writes the create before the first reference
            // and drops DestroyResource if the list is reset without executing.
            if (device_->state_bridge().is_resource_used_in_frame(id_) ||
                device_->state_bridge().is_resource_recorded(id_)) {
                device_->state_bridge().defer_destroy_resource(id_);
            } else if (!device_->release_deferred(id_)) {
                device_->state_bridge().destroy_or_defer_resource(id_, enc);
            }
            if (enc.size() != 0) {
                device_->submit(enc);
            }
        }
    }
    if (owned_guest_memory_ != 0) {
        com::guest_memory().release(owned_guest_memory_);
    }
    if (xmem_memory_ != nullptr) {
        pg6_xbox_XMemFree(xmem_memory_, kD3D12ResourceXMemAttributes);
    }
    device_->resources.fetch_sub(1);
}

uint64_t committed_memory_size(const GuestResourceDesc& d) {
    // The 64 KiB-block surface size where os/surface_blocks.h covers the resource, else the
    // size estimate (console layouts, including linear 0x100, use their GFX10 placement there).
    const DxgiFormatLayout l = dxgi_format_layout(d.Format);
    os::surface::Block64kLayout block{};
    if (d.Layout != gfx10::kConsoleLinearLayout && d.Dimension == ResourceDimension::Texture2D &&
        d.MipLevels <= 1 && d.DepthOrArraySize <= 1 &&
        d.SampleCount <= 1 && l.known && l.block_width == 1 && l.block_height == 1 && l.planes == 1 &&
        d.Width <= 0xFFFFFFFFull &&
        os::surface::block64k_layout_2d(static_cast<uint32_t>(d.Width), d.Height, l.block_bytes, &block)) {
        return block.size_bytes;
    }
    return resource_size_estimate(d);
}

uint32_t Resource::array_layers() const {
    return desc_.Dimension == ResourceDimension::Texture3D || is_buffer() ? 1u
                                                                          : std::max<uint32_t>(1, desc_.DepthOrArraySize);
}

uint32_t Resource::subresource_count() const {
    if (is_buffer()) {
        return 1;
    }
    const DxgiFormatLayout l = dxgi_format_layout(desc_.Format);
    return mips_ * array_layers() * (l.known ? l.planes : 1u);
}

bool Resource::initialize() {
    Device& dev = *device_;
    id_ = dev.allocate_id();
    if (is_buffer()) {
        const uint64_t size = desc_.Width;
        if (placement_.address != 0) {
            gpu_va_ = placement_.address;
            const uint64_t cpu = dev.cpu_address_for_gpu(placement_.address);
            // Inside an alias window or mirrored MEM_PHYSICAL reservation the CPU pages may be
            // mapped later (streaming): keep the CPU address; flush_shadow reads whatever is mapped
            // at each submission.
            if (cpu != placement_.address ||
                os::guest_window::runtime_range_host(cpu) != nullptr ||
                com::guest_to_host(cpu, static_cast<size_t>(size)) != nullptr) {
                cpu_va_ = cpu;
            } else {
                dev.note_bad_argument("resource at " + hex(placement_.address) +
                                      " is not guest memory: contents cannot be uploaded");
            }
        } else if (placement_.placed_heap) {
            gpu_va_ = placement_.placed_heap->base() + placement_.heap_offset;
            cpu_va_ = placement_.placed_heap->cpu_visible() ? gpu_va_ : 0;
        } else if (placement_.reserved || !heap_cpu_visible(placement_.heap)) {
            gpu_va_ = dev.allocate_gpu_va(size);
        } else {
            const uint64_t va = com::guest_memory().virtual_alloc(0, align_up(size, 64u << 10),
                                                                  os::kMemReserve | os::kMemCommit,
                                                                  os::PageProtect::ReadWrite);
            if (va == 0) {
                dev.note_bad_argument("out of guest memory for a " + std::to_string(size) + "-byte buffer");
                id_ = 0;
                return false;
            }
            owned_guest_memory_ = va;
            gpu_va_ = cpu_va_ = va;
        }
        // A small buffer is allocated with at least kSmallBufferPadding bytes: the title binds whole
        // 16-byte buffers (written by the GPU) as constant buffers to stages that declare uniform
        // blocks of up to a few KiB, and WebGPU needs the binding to cover the declared block. The
        // bytes past the guest's size are never written, so they stay zero (the zero-extension
        // WebGPU needs, not data; on the console the stage reads whatever memory follows).
        constexpr uint64_t kSmallBufferPadding = 4096;
        const uint64_t allocated = std::max<uint64_t>(align_up(size, 4), kSmallBufferPadding);
        CommandEncoder enc;
        enc.create_buffer(id_,
                          GpuBufferUsage::Vertex | GpuBufferUsage::Index | GpuBufferUsage::Uniform |
                              GpuBufferUsage::Storage | GpuBufferUsage::Indirect | GpuBufferUsage::CopyDst |
                              GpuBufferUsage::CopySrc,
                          allocated);
        dev.defer_create(id_, enc);  // written before the first packet that uses it (G66)
        dev.state_bridge().register_buffer(id_, gpu_va_, size, allocated);
        if (cpu_va_ != 0) {
            // O36: no host memory yet (see shadow_chunks_): the chunks come with the first uploads.
            shadow_size_ = size;
            shadow_chunks_.resize(static_cast<size_t>((size + kShadowChunk - 1) / kShadowChunk));
            dev.note_shadow_bytes(static_cast<int64_t>(size), 0);
            dev.track_cpu_buffer(this);
            tracked_ = true;
        }
    } else {
        mips_ = desc_.MipLevels == 0 ? full_mip_count(desc_) : desc_.MipLevels;
        const bool depth = (desc_.Flags & ResourceFlags::AllowDepthStencil) != 0;
        const bool rt = (desc_.Flags & ResourceFlags::AllowRenderTarget) != 0;
        if (rt && !depth && !placement_.placed_heap && !placement_.reserved && placement_.address == 0) {
            // The console's D3D12 allocates committed-resource memory with XMemAlloc, so a
            // title's XMem allocation hooks see it (and may place it in arenas of their own).
            // Only committed render targets for now: they are what the boot shows (other
            // resource kinds are left as they were until a title shows their accounting).
            const uint64_t size = committed_memory_size(desc_);
            xmem_memory_ = pg6_xbox_XMemAlloc(static_cast<size_t>(size), kD3D12ResourceXMemAttributes);
            if (xmem_memory_ == nullptr) {
                dev.note_bad_argument("XMemAlloc of " + std::to_string(size) + " bytes for a render target failed");
                id_ = 0;
                return false;
            }
        } else if (!depth && !placement_.placed_heap && !placement_.reserved && placement_.address == 0 &&
                   heap_cpu_visible(placement_.heap)) {
            const uint64_t size = align_up(std::max<uint64_t>(committed_memory_size(desc_), 1), 64u << 10);
            const uint64_t va = com::guest_memory().virtual_alloc(0, size, os::kMemReserve | os::kMemCommit,
                                                                  os::PageProtect::ReadWrite);
            if (va == 0) {
                dev.note_bad_argument("out of guest memory for a " + std::to_string(size) + "-byte texture");
                id_ = 0;
                return false;
            }
            owned_guest_memory_ = va;
        }
        // The texture's GPU VA is its memory (see the header comment).
        if (placement_.address != 0) {
            gpu_va_ = placement_.address;
            cpu_va_ = dev.cpu_address_for_gpu(gpu_va_);
        } else if (placement_.placed_heap) {
            gpu_va_ = placement_.placed_heap->base() + placement_.heap_offset;
            if (placement_.placed_heap->cpu_visible()) {
                cpu_va_ = gpu_va_;
            }
        } else if (xmem_memory_ != nullptr) {
            gpu_va_ = com::host_to_guest(xmem_memory_);
            cpu_va_ = dev.cpu_address_for_gpu(gpu_va_);
        } else if (owned_guest_memory_ != 0) {
            gpu_va_ = cpu_va_ = owned_guest_memory_;
        }
        if (gpu_va_ != 0 && xmem_memory_ == nullptr) {
            TextureProbe& p = texture_probe();
            std::lock_guard<std::mutex> lock(p.mutex);
            p.textures.push_back(this);
            probe_registered_ = true;
        }
        const uint32_t features = dev.state_bridge().options().features;
        const bool allow_uav = (desc_.Flags & ResourceFlags::AllowUnorderedAccess) != 0;
        const pso::FormatUsage usage = depth ? pso::FormatUsage::DepthStencil
                                             : (rt ? pso::FormatUsage::RenderTarget : pso::FormatUsage::Sampled);
        const pso::TextureFormatMapping m =
            pso::map_texture_format(static_cast<pso::DxgiFormat>(desc_.Format), usage, features);
        if (!m.supported()) {
            dev.note_bad_argument("texture format " + std::to_string(desc_.Format) +
                                  " has no WebGPU equivalent: resource created without a GPU texture");
        } else {
            GpuCreateTexture2Payload p{};
            p.id = id_;
            p.width = static_cast<uint32_t>(desc_.Width);
            p.height = std::max<uint32_t>(1, desc_.Height);
            p.depth_or_layers = std::max<uint32_t>(1, desc_.DepthOrArraySize);
            p.mip_levels = mips_;
            p.sample_count = pso::map_sample_count(desc_.SampleCount);
            p.dimension = static_cast<uint8_t>(desc_.Dimension == ResourceDimension::Texture1D
                                                   ? GpuTextureDimension::Dim1D
                                                   : (desc_.Dimension == ResourceDimension::Texture3D
                                                          ? GpuTextureDimension::Dim3D
                                                          : GpuTextureDimension::Dim2D));
            const pso::WgpuTextureFormat create_format =
                storage_base_format(m.format, allow_uav, p.sample_count, features);
            const bool can_store = allow_uav && p.sample_count == 1 && storage_capable(create_format, features);
            const bool can_render_uav = allow_uav && !can_store && p.sample_count == 1 &&
                                        p.dimension == static_cast<uint8_t>(GpuTextureDimension::Dim2D) &&
                                        renderable_uav_fallback(create_format, features);
            p.usage_flags = GpuTextureUsage::CopySrc | GpuTextureUsage::CopyDst;
            if ((desc_.Flags & ResourceFlags::DenyShaderResource) == 0) {
                p.usage_flags |= GpuTextureUsage::TextureBinding;
            }
            if ((rt || depth || can_render_uav) && p.dimension != static_cast<uint8_t>(GpuTextureDimension::Dim1D)) {
                p.usage_flags |= GpuTextureUsage::RenderAttachment;
            }
            if (allow_uav) {
                if (can_store) {
                    p.usage_flags |= GpuTextureUsage::StorageBinding;
                } else {
                    // WebGPU rejects the whole texture when STORAGE_BINDING is not allowed for its
                    // format; such a texture keeps its other usages (renderable 2D formats still
                    // fall back through a render-pass copy-out in state_bridge).
                    dev.note_bad_argument(std::string("UAV texture format ") + pso::texture_format_name(create_format) +
                                          " has no WebGPU storage binding: created without storage usage");
                }
            }
            p.format = static_cast<uint8_t>(create_format);
            // Views of a D3D12 resource may switch between the sRGB and the linear format (an
            // _SRGB texture rendered through a UNORM RTV, and the reverse): list the sibling.
            if (const pso::WgpuTextureFormat sibling = pso::srgb_sibling(create_format);
                sibling != pso::WgpuTextureFormat::Undefined) {
                p.view_format_count = 1;
                p.view_formats[0] = static_cast<uint8_t>(sibling);
            }
            CommandEncoder enc;
            enc.emit_packet(GpuOpcode::CreateTexture2, &p, sizeof(p));
            dev.defer_create(id_, enc);  // written before the first packet that uses it (G66)
            gpu_texture_ = true;
            // The address its descriptor words carry (descriptor_words.h): its GPU VA, or for a
            // texture without memory a range of the device's fake GPU VA space.
            descriptor_va_ = gpu_va_ != 0 ? gpu_va_
                                          : dev.allocate_gpu_va(std::max<uint64_t>(committed_memory_size(desc_), 1));
            bridge::TextureInfo info;
            info.format = static_cast<pso::DxgiFormat>(desc_.Format);
            info.width = p.width;
            info.height = p.height;
            info.depth_or_array_size = p.depth_or_layers;
            info.mip_levels = mips_;
            info.sample_count = p.sample_count;
            info.dimension = static_cast<GpuTextureDimension>(p.dimension);
            info.gpu_va = descriptor_va_;
            info.layout = desc_.Layout;
            info.storage = (p.usage_flags & GpuTextureUsage::StorageBinding) != 0;
            info.render_target = rt || can_render_uav;
            info.depth_stencil = depth || bridge::is_depth_stencil_dxgi_format(info.format);
            info.created_format = create_format;
            dev.state_bridge().register_texture(id_, info);
            va_watch::note_texture(descriptor_va_, committed_memory_size(desc_), p.width, p.height, desc_.Format, id_);
            if (const DxgiFormatLayout bl = dxgi_format_layout(desc_.Format); bl.known && bl.block_bytes != 0) {
                const bool is3d = desc_.Dimension == ResourceDimension::Texture3D;
                budget_bytes_ = texture_chain_bytes(p.width, p.height, is3d ? p.depth_or_layers : 1u, mips_,
                                                    bl.block_width, bl.block_height, bl.block_bytes, 0u,
                                                    is3d ? 1u : p.depth_or_layers);
                texture_budget().note_created(budget_bytes_);
                texture_budget().register_texture(id_, budget_bytes_, dev.present_epoch(),
                                                  /*pinned=*/rt || depth || back_buffer_.load(), /*evictable=*/false);
            }
            if (gpu_va_ != 0 && gpu_va_ < Device::kFakeGpuVaBase) {
                TrackedTexture tt{};
                tt.resource_id = id_;
                tt.streamable = placement_.reserved;
                tt.gpu_va = gpu_va_;
                tt.dxgi_format = desc_.Format;
                tt.width = p.width;
                tt.height = p.height;
                tt.depth = p.depth_or_layers;
                tt.mip_levels = mips_;
                tt.layout = desc_.Layout;
                tt.dimension = desc_.Dimension;
                tt.sample_count = p.sample_count;
                if (const DxgiFormatLayout bl2 = dxgi_format_layout(desc_.Format);
                    p.dimension == static_cast<uint8_t>(GpuTextureDimension::Dim2D) && p.depth_or_layers == 1u &&
                    p.sample_count == 1u && mips_ > 1u && bl2.known && bl2.block_width > 1u &&
                    budget_bytes_ > kSmallTextureMaxBytes &&
                    (p.usage_flags & (GpuTextureUsage::RenderAttachment | GpuTextureUsage::StorageBinding)) == 0) {
                    tt.has_create = true;  // the host allocation may be replaced at a capped size
                    tt.create = p;
                }
                dev.track_guest_texture(tt);
                texture_tracked_ = true;
            }
        }
    }
    d3d::ResourceStateDesc sd;
    sd.subresource_count = subresource_count();
    sd.initial_state = initial_state_;
    sd.implicit_promotion = is_buffer() || (desc_.Flags & ResourceFlags::AllowSimultaneousAccess) != 0;
    dev.resource_states().register_resource(id_, sd);
    count_work(Work::ResourcesCreated);
    return true;
}

TextureProbeTotals probe_texture_contents(size_t budget) {
    constexpr uint64_t kProbeBytes = 64u << 10;
    TextureProbe& p = texture_probe();
    std::lock_guard<std::mutex> lock(p.mutex);
    for (size_t n = 0; n < budget && !p.textures.empty(); ++n) {
        p.cursor %= p.textures.size();
        Resource* r = p.textures[p.cursor++];
        if (r->probe_seen_data_) {
            continue;
        }
        const uint64_t len = std::min<uint64_t>(kProbeBytes, std::max<uint64_t>(1, resource_size_estimate(r->desc_)));
        // GPU-window addresses read their bytes from the CPU window (Device::register_gpu_alias).
        const uint64_t cpu = r->device_->cpu_address_for_gpu(r->gpu_va_);
        const com::GuestRangeView view(cpu, static_cast<size_t>(len));
        const uint8_t* bytes = view.data();
        if (bytes == nullptr || all_zero_bytes(bytes, static_cast<size_t>(len))) {
            continue;
        }
        r->probe_seen_data_ = true;
        ++p.with_data;
        if (p.with_data <= 64) {  // the log names resource textures; view-only ones are counted only
            const GuestResourceDesc& d = r->desc_;
            char line[256];
            std::snprintf(line, sizeof(line),
                          "d3d12_xs: texture data written directly: format %u, %" PRIu64 "x%ux%u, %u mip(s), "
                          "layout 0x%x, flags 0x%x, dimension %u (texture %u of %zu registered)",
                          d.Format, d.Width, d.Height, static_cast<unsigned>(d.DepthOrArraySize), r->mips_,
                          d.Layout, d.Flags, d.Dimension, static_cast<unsigned>(p.with_data), p.textures.size());
            os::log(line);
        }
    }
    for (size_t n = 0; n < budget && !p.views.empty(); ++n) {
        p.view_cursor %= p.views.size();
        ViewProbe& v = p.views[p.view_cursor++];
        if (v.seen_data) {
            continue;
        }
        const com::GuestRangeView view(v.cpu, static_cast<size_t>(v.len));
        const uint8_t* bytes = view.data();
        if (bytes == nullptr || all_zero_bytes(bytes, static_cast<size_t>(v.len))) {
            continue;
        }
        v.seen_data = true;
        ++p.views_with_data;
    }
    TextureProbeTotals t;
    t.views_registered = p.views.size();
    t.views_with_data = p.views_with_data;
    t.registered = p.textures.size() + t.views_registered;
    t.with_data = p.with_data + t.views_with_data;
    return t;
}

void register_view_texture_probe(uint64_t cpu_address, uint64_t size) {
    if (cpu_address == 0) {
        return;
    }
    TextureProbe& p = texture_probe();
    std::lock_guard<std::mutex> lock(p.mutex);
    // O36: one probe per address (an evicted-and-recreated texture reuses it), and the table is capped.
    for (ViewProbe& existing : p.views) {
        if (existing.cpu == cpu_address) {
            return;
        }
    }
    constexpr size_t kMaxViewProbes = 4096;
    if (p.views.size() >= kMaxViewProbes) {
        const size_t drop = kMaxViewProbes / 4;
        for (size_t i = 0; i < drop; ++i) {
            if (p.views[i].seen_data && p.views_with_data > 0) {
                --p.views_with_data;
            }
        }
        p.views.erase(p.views.begin(), p.views.begin() + static_cast<std::ptrdiff_t>(drop));
        p.view_cursor = 0;
    }
    p.views.push_back(ViewProbe{cpu_address, std::min<uint64_t>(64u << 10, std::max<uint64_t>(1, size)), false});
}

void Resource::mark_back_buffer() {
    bridge::TextureInfo info;
    info.format = static_cast<pso::DxgiFormat>(desc_.Format);
    info.width = static_cast<uint32_t>(desc_.Width);
    info.height = std::max<uint32_t>(1, desc_.Height);
    info.depth_or_array_size = 1;
    info.mip_levels = mips_;
    info.sample_count = 1;
    info.dimension = GpuTextureDimension::Dim2D;
    info.swapchain = true;
    info.gpu_va = descriptor_va_;
    device_->state_bridge().register_texture(id_, info);
    back_buffer_.store(true);
    texture_budget().set_texture_pinned(id_, true);
}

bool Resource::shadow_same(uint64_t off, const uint8_t* live, uint64_t n) const {
    const uint8_t* chunk = shadow_chunks_[static_cast<size_t>(off / kShadowChunk)].get();
    if (chunk == nullptr) {
        return all_zero_bytes(live, static_cast<size_t>(n));
    }
    return same_bytes(chunk + off % kShadowChunk, live, static_cast<size_t>(n));
}

void Resource::shadow_store(uint64_t off, const uint8_t* src, uint64_t n) {
    while (n != 0) {
        const size_t index = static_cast<size_t>(off / kShadowChunk);
        const uint64_t in_chunk = off % kShadowChunk;
        const uint64_t take = std::min(n, kShadowChunk - in_chunk);
        std::unique_ptr<uint8_t[]>& chunk = shadow_chunks_[index];
        if (chunk == nullptr && !all_zero_bytes(src, static_cast<size_t>(take))) {
            const uint64_t chunk_bytes = std::min(kShadowChunk, shadow_size_ - index * kShadowChunk);
            chunk = std::make_unique<uint8_t[]>(static_cast<size_t>(chunk_bytes));  // zeroed
            shadow_resident_ += chunk_bytes;
            device_->note_shadow_bytes(0, static_cast<int64_t>(chunk_bytes));
        }
        if (chunk != nullptr) {
            std::memcpy(chunk.get() + in_chunk, src, static_cast<size_t>(take));
        }
        off += take;
        src += take;
        n -= take;
    }
}

uint64_t Resource::flush_shadow(CommandEncoder& encoder, os::guest_window::MarkPass* marks) {
    if (cpu_va_ == 0 || shadow_size_ == 0) {
        return 0;
    }
    const uint64_t size = shadow_size_;
    const size_t num_pages = static_cast<size_t>((size + kShadowPage - 1) / kShadowPage);
    const bool verify_marks = os::guest_window::write_marks_verified();
    // Held for the whole compare / upload-copy below so a concurrent decommit cannot fault it.
    const com::GuestRangeView view(cpu_va_, static_cast<size_t>(size));
    const uint8_t* live = view.data();
    std::vector<uint8_t> copy;
    std::unique_ptr<uint8_t[]> sparse_copy;
    // Keep the bulk staging allocation alive past the small candidate list, as before:
    // their destruction order affects allocator reuse on subsequent dense flushes.
    std::vector<uint32_t> dirty_page_indices;
    bool has_candidates = false;
    bool window_candidates = false;
    bool window_prepared = false;
    auto finalize_dirty_pages = [&] {
        if (dirty_page_indices.size() > 1) {
            std::sort(dirty_page_indices.begin(), dirty_page_indices.end());
            dirty_page_indices.erase(std::unique(dirty_page_indices.begin(), dirty_page_indices.end()),
                                     dirty_page_indices.end());
        }
    };
    auto prepare_window = [&] {
        window_prepared = true;
        std::vector<uint64_t> slots;
        const bool marked = marks != nullptr ? marks->take(cpu_va_, static_cast<size_t>(size), &slots)
                                             : os::guest_window::take_written(cpu_va_, static_cast<size_t>(size), &slots);
        if (marked && window_marked_) {
            has_candidates = true;
            window_candidates = true;
            dirty_page_indices.clear();
            for (uint64_t off : slots) {
                const uint64_t slot_end = ((cpu_va_ + off) | (os::guest_window::kSlotBytes - 1)) + 1 - cpu_va_;
                const uint64_t hi = std::min<uint64_t>(size, slot_end);
                for (uint64_t page = off / kShadowPage * kShadowPage; page < hi; page += kShadowPage) {
                    const size_t pi = static_cast<size_t>(page / kShadowPage);
                    if (pi < num_pages) {
                        dirty_page_indices.push_back(static_cast<uint32_t>(pi));
                    }
                }
            }
            finalize_dirty_pages();
        }
        window_marked_ = marked;
    };
    if (live == nullptr && window_marked_ && !verify_marks && marks != nullptr &&
        marks->at_most_half_marked(cpu_va_, static_cast<size_t>(size))) {
        prepare_window();
        if (window_candidates && !verify_marks && dirty_page_indices.size() < num_pages) {
            if (dirty_page_indices.empty()) {
                write_tracked_ = false;
                tracked_host_ = 0;
                device_->note_shadow_compared(0);
                count_work(Work::ShadowComparedBytes, 0);
                return 0;
            }
            // Keep the bulk copy when every page is a candidate. Otherwise only candidate
            // pages are subsequently read or uploaded. Avoid initializing and
            // copying an entire fragmented buffer for a single dirty window slot.
            sparse_copy = std::make_unique_for_overwrite<uint8_t[]>(static_cast<size_t>(size));
            for (uint32_t pi : dirty_page_indices) {
                const uint64_t off = static_cast<uint64_t>(pi) * kShadowPage;
                const size_t n = static_cast<size_t>(std::min(kShadowPage, size - off));
                uint8_t* dst = sparse_copy.get() + off;
                const uint8_t* shadow = shadow_chunks_[off / kShadowChunk].get();
                if (shadow != nullptr) {
                    std::memcpy(dst, shadow + off % kShadowChunk, n);
                } else {
                    std::memset(dst, 0, n);
                }
                // Unreadable pages retain the prior shadow, including any readable prefix.
                com::read_guest(cpu_va_ + off, dst, n);
            }
            live = sparse_copy.get();
        }
    }
    if (live == nullptr) {
        // Not mapped in one piece (e.g. an alias-window buffer whose pages are mapped one by one):
        // unreadable pages keep their shadow contents, i.e. count as unchanged.
        copy.assign(static_cast<size_t>(size), 0);
        for (size_t i = 0; i < shadow_chunks_.size(); ++i) {
            if (shadow_chunks_[i] != nullptr) {
                const uint64_t off = i * kShadowChunk;
                std::memcpy(copy.data() + off, shadow_chunks_[i].get(),
                            static_cast<size_t>(std::min(kShadowChunk, size - off)));
            }
        }
        if (!com::read_guest(cpu_va_, copy.data(), copy.size())) {
            for (uint64_t page = 0; page < size; page += kShadowPage) {
                const uint64_t n = std::min(kShadowPage, size - page);
                com::read_guest(cpu_va_ + page, copy.data() + page, static_cast<size_t>(n));
            }
        }
        live = copy.data();
    }
    // Written pages since the previous flush (where the arena tracks writes): only those can differ
    // from the shadow. The tracking is reset before the comparison, so a write racing with it is
    // reported again at the next flush.
    std::vector<uint64_t> written;
    const bool tracked = sparse_copy == nullptr && copy.empty() && os::take_written_pages(live, size, &write_cursor_, &written);
    const uint64_t skew = reinterpret_cast<uintptr_t>(live) & (os::kWrittenPageSize - 1);
    if (tracked && write_tracked_) {
        has_candidates = true;
        dirty_page_indices.clear();
        if (!written.empty()) {
            dirty_page_indices.reserve(written.size() * (skew != 0 ? size_t{2} : size_t{1}));
            for (uint64_t off : written) {
                const uint64_t lo = off > skew ? off - skew : 0;
                const uint64_t hi = std::min<uint64_t>(size, off + os::kWrittenPageSize - skew);
                for (uint64_t page = lo / kShadowPage * kShadowPage; page < hi; page += kShadowPage) {
                    const size_t pi = static_cast<size_t>(page / kShadowPage);
                    if (pi < num_pages) {
                        dirty_page_indices.push_back(static_cast<uint32_t>(pi));
                    }
                }
            }
            finalize_dirty_pages();
        }
    }
    write_tracked_ = tracked;
    tracked_host_ = tracked ? reinterpret_cast<uintptr_t>(live) : 0;
    // A fragmented view already took the shared marks before staging its candidate pages.
    if (!tracked && !window_prepared && os::guest_window::translated(cpu_va_)) {
        prepare_window();
    }
    // Verification: compare the unmarked pages too and count changes found there (a writer that
    // did not mark); such a page is uploaded all the same.
    const bool verify = window_candidates && verify_marks;
    uint64_t run_start = 0;
    bool in_run = false;
    uint64_t uploaded = 0;
    auto emit = [&](uint64_t begin, uint64_t end) {
        va_watch::touch("upload (CPU-visible buffer flush)", gpu_va_ + begin, end - begin,
                        "buffer id " + std::to_string(id_));
        for (uint64_t off = begin; off < end; off += kMaxWriteChunk) {
            const uint64_t n = std::min(kMaxWriteChunk, end - off);
            count_work(Work::BufferUploads);
            encoder.write_buffer(id_, static_cast<uint32_t>(off), live + off, static_cast<uint32_t>(n));
            shadow_store(off, live + off, n);
            uploaded += n;
        }
    };
    if (has_candidates && !verify) {
        // O(dirty_pages): iterate only the sorted unique candidate pages directly instead of
        // scanning a full bitmap across every page of the buffer.
        uint64_t run_end = 0;
        uint64_t compared = 0;
        for (uint32_t pi : dirty_page_indices) {
            const uint64_t page = static_cast<uint64_t>(pi) * kShadowPage;
            const uint64_t n = std::min(kShadowPage, size - page);
            compared += n;
            if (shadow_same(page, live + page, n)) {
                continue;
            }
            if (in_run && run_end == page) {
                run_end = page + n;
                continue;
            }
            if (in_run) {
                emit(run_start, run_end);
            }
            run_start = page;
            run_end = page + n;
            in_run = true;
        }
        if (in_run) {
            emit(run_start, run_end);
        }
        device_->note_shadow_compared(compared);
        count_work(Work::ShadowComparedBytes, compared);
        if (uploaded != 0) {
            device_->note_upload(uploaded);
        }
        return uploaded;
    }
    // Every page is compared here (no candidates, or verification): the bulk of the work when the
    // buffer has no write tracking (R40: split across helper threads, compare_pages).
    std::vector<uint8_t> differs;
    compare_pages(num_pages, size,
                  [&](size_t i) {
                      const uint64_t page = static_cast<uint64_t>(i) * kShadowPage;
                      return shadow_same(page, live + page, std::min(kShadowPage, size - page));
                  },
                  &differs);
    device_->note_shadow_compared(size);
    count_work(Work::ShadowComparedBytes, size);
    count_work(Work::ShadowFullCompares);
    count_work(Work::ShadowFullCompareBytes, size);
    size_t dirty_cursor = 0;
    for (uint64_t page = 0; page < size; page += kShadowPage) {
        const size_t pi = static_cast<size_t>(page / kShadowPage);
        bool listed = !has_candidates;
        if (has_candidates) {
            while (dirty_cursor < dirty_page_indices.size() &&
                   static_cast<size_t>(dirty_page_indices[dirty_cursor]) < pi) {
                ++dirty_cursor;
            }
            listed = dirty_cursor < dirty_page_indices.size() &&
                     static_cast<size_t>(dirty_page_indices[dirty_cursor]) == pi;
        }
        const bool changed = differs[pi] != 0;
        if (changed && !listed && !os::guest_window::write_marked(cpu_va_ + page)) {
            // (A slot marked by now was written while this flush compared: not missed, and its
            // mark reports it again.)
            os::guest_window::note_verify_miss(cpu_va_ + page);
        }
        if (changed && !in_run) {
            run_start = page;
            in_run = true;
        } else if (!changed && in_run) {
            emit(run_start, page);
            in_run = false;
        }
    }
    if (in_run) {
        emit(run_start, size);
    }
    if (uploaded != 0) {
        device_->note_upload(uploaded);
    }
    return uploaded;
}

HResult Resource::Map(uint32_t subresource, uint64_t, OutPtr data) {
    // Textures: on the console a texture's memory is ordinary CPU-addressable memory, and Map of
    // subresource 0 returns its base. Titles add the subresource offsets they take from the XG
    // layout themselves and write texels in that layout; the dirty-tracked texture-memory upload
    // (sync_bound_texture_memory) carries those writes to the GPU. Other subresources are refused
    // and reported rather than guessed.
    if (!is_buffer() && cpu_va_ != 0 && subresource != 0) {
        device_->note_bad_argument("ID3D12Resource::Map of texture subresource " + std::to_string(subresource) +
                                   " (id " + std::to_string(id_) + "): only subresource 0 is mapped");
        if (data) {
            data.write(0);
        }
        return kEInvalidArg;
    }
    if (cpu_va_ == 0 || subresource != 0) {
        device_->note_bad_argument("ID3D12Resource::Map on a resource without CPU access (id " + std::to_string(id_) +
                                   ")");
        if (data) {
            data.write(0);
        }
        return kEInvalidArg;
    }
    if (data) {
        data.write(cpu_va_);
    }
    return kSOk;
}

void Resource::Unmap(uint32_t, uint64_t) {
    // Nothing to do: CPU-visible buffers are compared with their shadow at ExecuteCommandLists.
}

HResult Resource::WriteToSubresource(com::CallContext& ctx) {
    // (DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch)
    const uint32_t sub = static_cast<uint32_t>(ctx.int_arg(1));
    const uint64_t box_va = ctx.int_arg(2);
    const uint64_t src = ctx.int_arg(3);
    const uint32_t row_pitch = static_cast<uint32_t>(ctx.int_arg(4));
    const uint32_t depth_pitch = static_cast<uint32_t>(ctx.int_arg(5));
    if (is_buffer() || !gpu_texture_ || sub >= subresource_count() || src == 0) {
        device_->note_bad_argument("WriteToSubresource: unsupported target or arguments");
        return kEInvalidArg;
    }
    const uint32_t mip = sub % mips_;
    const uint32_t layer = (sub / mips_) % array_layers();
    uint32_t host_mip = mip;
    if (!device_->state_bridge().translate_host_mip(id_, &host_mip)) {
        return kSOk;  // a level the capped host texture lacks (gpu/texture_budget.h): lower resolution
    }
    const uint32_t w = std::max<uint32_t>(1, static_cast<uint32_t>(desc_.Width >> mip));
    const uint32_t h = std::max<uint32_t>(1, desc_.Height >> mip);
    const uint32_t d = desc_.Dimension == ResourceDimension::Texture3D ? std::max<uint32_t>(1, desc_.DepthOrArraySize >> mip) : 1u;
    GuestBox box{0, 0, 0, w, h, d};
    if (box_va != 0 && !com::read_guest(box_va, box)) {
        device_->note_bad_argument("WriteToSubresource: unreadable box");
        return kEInvalidArg;
    }
    const bool depth = (desc_.Flags & ResourceFlags::AllowDepthStencil) != 0;
    const bool rt = (desc_.Flags & ResourceFlags::AllowRenderTarget) != 0;
    const pso::FormatUsage usage = depth ? pso::FormatUsage::DepthStencil
                                         : (rt ? pso::FormatUsage::RenderTarget : pso::FormatUsage::Sampled);
    const pso::TextureFormatMapping mapped =
        pso::map_texture_format(static_cast<pso::DxgiFormat>(desc_.Format), usage, device_->state_bridge().options().features);
    const CpuWidenSpec widen_spec =
        mapped.supported() ? cpu_widen_spec(static_cast<pso::DxgiFormat>(desc_.Format), mapped.format) : CpuWidenSpec{};
    if (widen_spec.src_bpp == 0 && device_->upload_needs_conversion(desc_.Format, usage)) {
        device_->note_unsupported_copy("WriteToSubresource: DXGI format " + std::to_string(desc_.Format) +
                                       " needs CPU conversion for its WebGPU format (not implemented); skipped");
        return kSOk;
    }
    va_watch_resource("upload (WriteToSubresource)", this, "subresource " + std::to_string(sub));
    const DxgiFormatLayout l = dxgi_format_layout(desc_.Format);
    const uint32_t raw_bw = std::max<uint32_t>(1, box.right - box.left);
    const uint32_t raw_bh = std::max<uint32_t>(1, box.bottom - box.top);
    const uint32_t bd = std::max<uint32_t>(1, box.back - box.front);
    // WebGPU requires block-compressed copySize and origin to be multiples of the block dimensions
    // (small mips like 2x2 / 1x1 use their 4x4 physical block size).
    const uint32_t bw = l.block_width > 1 ? static_cast<uint32_t>(align_up(raw_bw, l.block_width)) : raw_bw;
    const uint32_t bh = l.block_height > 1 ? static_cast<uint32_t>(align_up(raw_bh, l.block_height)) : raw_bh;
    const uint32_t origin_x = l.block_width > 1 ? (box.left / l.block_width) * l.block_width : box.left;
    const uint32_t origin_y = l.block_height > 1 ? (box.top / l.block_height) * l.block_height : box.top;
    const uint32_t rows = (bh + l.block_height - 1) / l.block_height;
    const uint64_t len = uint64_t{depth_pitch} * (bd - 1) + uint64_t{row_pitch} * rows;
    std::vector<uint8_t> bytes(static_cast<size_t>(len));
    if (!com::read_guest(src, bytes.data(), bytes.size())) {
        device_->note_bad_argument("WriteToSubresource: unreadable source");
        return kEInvalidArg;
    }
    CommandEncoder enc;
    const bool is3d = desc_.Dimension == ResourceDimension::Texture3D;
    count_work(Work::TextureUploads);
    if (widen_spec.src_bpp != 0) {
        std::vector<uint8_t> wide;
        uint32_t pitch = 0;
        if (!widen_promoted_rows(static_cast<pso::DxgiFormat>(desc_.Format), mapped.format, bytes.data(), bytes.size(),
                                 bw, rows, bd, row_pitch, depth_pitch, &wide, &pitch)) {
            device_->note_bad_argument("WriteToSubresource: widened source smaller than its box");
            return kEInvalidArg;
        }
        enc.write_texture(id_, host_mip, origin_x, origin_y, is3d ? box.front : layer, bw, bh, bd, pitch, rows,
                          wide.data(), static_cast<uint32_t>(wide.size()));
        device_->submit(enc);
        device_->note_upload(wide.size());
        device_->note_texture_upload();
        return kSOk;
    }
    enc.write_texture(id_, host_mip, origin_x, origin_y, is3d ? box.front : layer, bw, bh, bd, row_pitch,
                      row_pitch != 0 ? depth_pitch / row_pitch : rows, bytes.data(), static_cast<uint32_t>(len));
    device_->submit(enc);
    device_->note_upload(len);
    device_->note_texture_upload();
    return kSOk;
}

HResult Resource::ReadFromSubresource(com::CallContext&) {
    device_->note_unsupported_copy("ReadFromSubresource: GPU -> CPU readback has no packet");
    return kENotImpl;
}

HResult Resource::GetHeapProperties(GuestPtr<GuestHeapProperties> properties, GuestPtr<uint32_t> flags) {
    if (placement_.reserved) {
        return kEInvalidArg;  // D3D12: reserved resources have no heap
    }
    GuestHeapProperties p = placement_.placed_heap ? placement_.placed_heap->desc().Properties : placement_.heap;
    if (properties) {
        properties.write(p);
    }
    if (flags) {
        flags.write(placement_.placed_heap ? placement_.placed_heap->desc().Flags : placement_.heap_flags);
    }
    return kSOk;
}

ResourceDesc1Result Resource::GetDesc1() {
    ResourceDesc1Result r{};
    std::memcpy(r.bytes, &desc_, 52);  // the shared D3D12_RESOURCE_DESC prefix; mip region 0
    return r;
}

// ---------------------------------------------------------------------------------------------
// Device: resources, heaps
// ---------------------------------------------------------------------------------------------

namespace {
// Xbox: a texture's layout field may also hold 0x100 | an XG swizzle mode. The executable builds
// 0x100, 0x102, 0x10A and 0x100 | a per-texture mode byte itself, and XGSuggestSwizzleMode writes
// into that field (kXgDefaultSwizzleMode). The guest layer keeps no CPU image of textures, so the
// mode does not change what is created; it is accepted, not interpreted.
bool xbox_swizzle_layout(uint32_t layout) { return (layout & ~0xFFu) == 0x100u; }

bool validate_desc(Device& dev, const GuestResourceDesc& d) {
    const bool layout_ok =
        d.Layout <= 3 || (d.Dimension != ResourceDimension::Buffer && xbox_swizzle_layout(d.Layout));
    if (d.Dimension < ResourceDimension::Buffer || d.Dimension > ResourceDimension::Texture3D || d.Width == 0 ||
        !layout_ok) {
        dev.note_bad_argument("invalid resource description (dimension " + std::to_string(d.Dimension) + ", width " +
                              std::to_string(d.Width) + ", layout " + std::to_string(d.Layout) + ")");
        return false;
    }
    if (d.Dimension == ResourceDimension::Buffer && (d.Format != 0 || d.Height > 1)) {
        dev.note_bad_argument("invalid buffer description (format " + std::to_string(d.Format) + ")");
        return false;
    }
    if (d.Dimension != ResourceDimension::Buffer &&
        (d.Width > 0x10000 || d.Height == 0 || d.DepthOrArraySize == 0 || !dxgi_format_layout(d.Format).known)) {
        dev.note_bad_argument("invalid texture description (" + std::to_string(d.Width) + "x" +
                              std::to_string(d.Height) + "x" + std::to_string(d.DepthOrArraySize) + ", format " +
                              std::to_string(d.Format) + ")");
        return false;
    }
    return true;
}
}  // namespace

HResult Device::create_resource(const GuestResourceDesc& desc, const GuestHeapProperties& heap, uint32_t heap_flags,
                                uint32_t initial_state, com::Ref<Heap> placed_heap, uint64_t heap_offset,
                                uint64_t address, bool reserved, const Guid& riid, OutPtr out) {
    if (!validate_desc(*this, desc)) {
        return kEInvalidArg;
    }
    if (placed_heap && heap_offset + (desc.Dimension == ResourceDimension::Buffer ? desc.Width : 0) >
                           placed_heap->desc().SizeInBytes) {
        note_bad_argument("placed resource exceeds its heap");
        return kEInvalidArg;
    }
    Resource::Placement pl;
    pl.heap = heap;
    pl.heap_flags = heap_flags;
    pl.placed_heap = std::move(placed_heap);
    pl.heap_offset = heap_offset;
    pl.address = address;
    pl.reserved = reserved;
    auto resource =
        com::create_object<Resource>(com::implements(kID3D12Resource), device_ref(this), desc, std::move(pl), initial_state);
    if (resource && !resource->initialize()) {
        return kEOutOfMemory;
    }
    return hand_out(std::move(resource), riid, out);
}

com::Ref<Resource> Device::create_internal_buffer(uint64_t size) {
    GuestResourceDesc desc{};
    desc.Dimension = ResourceDimension::Buffer;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleCount = 1;
    desc.Layout = 1;  // ROW_MAJOR
    Resource::Placement pl;
    pl.heap.Type = HeapType::Default;
    auto resource = com::create_object<Resource>(com::implements(kID3D12Resource), device_ref(this), desc, std::move(pl),
                                                 uint32_t{0});
    if (!resource || !resource->initialize()) {
        return {};
    }
    return resource;
}

HResult Device::CreateCommittedResource(com::CallContext& ctx) {
    // (pHeapProperties, HeapFlags, pDesc, InitialResourceState, pOptimizedClearValue, riid, ppvResource)
    GuestHeapProperties props{};
    GuestResourceDesc desc{};
    Guid riid{};
    if (!com::read_guest(ctx.int_arg(1), props) || !com::read_guest(ctx.int_arg(3), desc) ||
        !com::read_guest(ctx.int_arg(6), riid)) {
        note_bad_argument("CreateCommittedResource: unreadable arguments");
        return kEInvalidArg;
    }
    return create_resource(desc, props, static_cast<uint32_t>(ctx.int_arg(2)), static_cast<uint32_t>(ctx.int_arg(4)),
                           {}, 0, 0, false, riid, OutPtr(ctx.int_arg(7)));
}

HResult Device::CreateCommittedResource3(com::CallContext& ctx) {
    // (pHeapProperties, HeapFlags, pDesc (DESC1), InitialLayout, pOptimizedClearValue,
    //  pProtectedSession, NumCastableFormats, pCastableFormats, riid, ppvResource)
    GuestHeapProperties props{};
    GuestResourceDesc desc{};
    Guid riid{};
    if (!com::read_guest(ctx.int_arg(1), props) || !com::read_guest(ctx.int_arg(3), &desc, 52) ||
        !com::read_guest(ctx.int_arg(9), riid)) {
        note_bad_argument("CreateCommittedResource (Xbox slot 127): unreadable arguments");
        return kEInvalidArg;
    }
    // Enhanced-barrier layouts start every resource in COMMON for the state tracker.
    return create_resource(desc, props, static_cast<uint32_t>(ctx.int_arg(2)), d3d::ResourceStates::Common, {}, 0, 0,
                           false, riid, OutPtr(ctx.int_arg(10)));
}

HResult Device::CreateHeap(const GuestHeapDesc* desc, const Guid& riid, OutPtr out) {
    if (desc == nullptr || desc->SizeInBytes == 0 || desc->Properties.Type < 1 || desc->Properties.Type > 4) {
        note_bad_argument("CreateHeap: invalid descriptor");
        return kEInvalidArg;
    }
    auto heap = com::create_object<Heap>(com::implements(kID3D12Heap), device_ref(this), *desc);
    if (heap && !heap->valid()) {
        note_bad_argument("CreateHeap: out of guest memory for " + std::to_string(desc->SizeInBytes) + " bytes");
        return kEOutOfMemory;
    }
    return hand_out(std::move(heap), riid, out);
}

HResult Device::CreatePlacedResource(com::CallContext& ctx) {
    // (pHeap, HeapOffset, pDesc, InitialState, pOptimizedClearValue, riid, ppvResource)
    auto heap_arg = InterfaceArg<IHeapHost>::resolve(ctx.int_arg(1));
    GuestResourceDesc desc{};
    Guid riid{};
    if (!heap_arg || !com::read_guest(ctx.int_arg(3), desc) || !com::read_guest(ctx.int_arg(6), riid)) {
        note_bad_argument("CreatePlacedResource: invalid heap or unreadable arguments");
        return kEInvalidArg;
    }
    auto heap = com::Ref<Heap>::share(static_cast<Heap*>(heap_arg.get()));
    return create_resource(desc, GuestHeapProperties{}, 0, static_cast<uint32_t>(ctx.int_arg(4)), std::move(heap),
                           ctx.int_arg(2), 0, false, riid, OutPtr(ctx.int_arg(7)));
}

HResult Device::CreatePlacedResourceX(com::CallContext& ctx) {
    // Xbox slot 133: (address, pDesc, InitialState, pOptimizedClearValue, riid, ppvResource)
    const uint64_t address = ctx.int_arg(1);
    GuestResourceDesc desc{};
    Guid riid{};
    if (address == 0 || !com::read_guest(ctx.int_arg(2), &desc, 52) || !com::read_guest(ctx.int_arg(5), riid)) {
        note_bad_argument("CreatePlacedResource (Xbox slot 133): unreadable arguments");
        return kEInvalidArg;
    }
    GuestHeapProperties props{};
    props.Type = HeapType::Custom;
    props.CPUPageProperty = 2;
    return create_resource(desc, props, 0, static_cast<uint32_t>(ctx.int_arg(3)), {}, 0, address, false, riid,
                           OutPtr(ctx.int_arg(6)));
}

HResult Device::CreateReservedResource(uint64_t desc_va, uint32_t state, uint64_t, const Guid& riid, OutPtr out) {
    GuestResourceDesc desc{};
    if (!com::read_guest(desc_va, desc)) {
        note_bad_argument("CreateReservedResource: unreadable descriptor");
        return kEInvalidArg;
    }
    return create_resource(desc, GuestHeapProperties{}, 0, state, {}, 0, 0, true, riid, out);
}

// ---------------------------------------------------------------------------------------------
// Device: views
// ---------------------------------------------------------------------------------------------

namespace {
uint32_t resource_id(const InterfaceArg<IResourceHost>& r) {
    return r ? static_cast<Resource*>(r.get())->id() : 0u;
}
}  // namespace

bool parse_srv_desc(const GuestShaderResourceViewDesc& g, d3d::ViewDesc* out) {
    d3d::ViewDesc vd;
    vd.format = g.Format;
    vd.dimension = g.ViewDimension;
    vd.component_mapping = g.Shader4ComponentMapping;
    const uint32_t* u = g.u;
    switch (g.ViewDimension) {
    case d3d::ViewDimension::Buffer:
        vd.first_element = u64_at(u, 0);
        vd.num_elements = u[2];
        vd.structure_byte_stride = u[3];
        vd.flags = u[4];
        break;
    case d3d::ViewDimension::Texture1D:
    case d3d::ViewDimension::Texture3D:
    case d3d::ViewDimension::TextureCube:
        vd.first_mip = u[0];
        vd.mip_levels = mips_or_all(u[1]);
        vd.min_lod_clamp = f32(u[2]);
        break;
    case d3d::ViewDimension::Texture1DArray:
    case d3d::ViewDimension::TextureCubeArray:
        vd.first_mip = u[0];
        vd.mip_levels = mips_or_all(u[1]);
        vd.first_array_slice = u[2];
        vd.array_size = u[3];
        vd.min_lod_clamp = f32(u[4]);
        break;
    case d3d::ViewDimension::Texture2D:
        vd.first_mip = u[0];
        vd.mip_levels = mips_or_all(u[1]);
        vd.plane_slice = u[2];
        vd.min_lod_clamp = f32(u[3]);
        break;
    case d3d::ViewDimension::Texture2DArray:
        vd.first_mip = u[0];
        vd.mip_levels = mips_or_all(u[1]);
        vd.first_array_slice = u[2];
        vd.array_size = u[3];
        vd.plane_slice = u[4];
        vd.min_lod_clamp = f32(u[5]);
        break;
    case d3d::ViewDimension::Texture2DMs:
        break;
    case d3d::ViewDimension::Texture2DMsArray:
        vd.first_array_slice = u[0];
        vd.array_size = u[1];
        break;
    case 11:  // RAYTRACING_ACCELERATION_STRUCTURE: Location
        vd.first_element = u64_at(u, 0);
        break;
    default:
        return false;
    }
    *out = vd;
    return true;
}

bool parse_uav_desc(const GuestUnorderedAccessViewDesc& g, d3d::ViewDesc* out) {
    d3d::ViewDesc vd;
    vd.format = g.Format;
    vd.dimension = g.ViewDimension;
    const uint32_t* u = g.u;
    switch (g.ViewDimension) {
    case d3d::ViewDimension::Buffer:
        vd.first_element = u64_at(u, 0);
        vd.num_elements = u[2];
        vd.structure_byte_stride = u[3];
        vd.counter_offset_in_bytes = u64_at(u, 4);
        vd.flags = u[6];
        break;
    case d3d::ViewDimension::Texture1D:
        vd.first_mip = u[0];
        break;
    case d3d::ViewDimension::Texture1DArray:
    case d3d::ViewDimension::Texture3D:
        vd.first_mip = u[0];
        vd.first_array_slice = u[1];
        vd.array_size = u[2];
        break;
    case d3d::ViewDimension::Texture2D:
        vd.first_mip = u[0];
        vd.plane_slice = u[1];
        break;
    case d3d::ViewDimension::Texture2DArray:
        vd.first_mip = u[0];
        vd.first_array_slice = u[1];
        vd.array_size = u[2];
        vd.plane_slice = u[3];
        break;
    case d3d::ViewDimension::Texture2DMs:
        break;
    case d3d::ViewDimension::Texture2DMsArray:
        vd.first_array_slice = u[0];
        vd.array_size = u[1];
        break;
    default:
        return false;
    }
    *out = vd;
    return true;
}

void Device::CreateShaderResourceView(InterfaceArg<IResourceHost> resource, uint64_t desc_va, uint64_t dest) {
    d3d::ViewDesc vd;
    const d3d::ViewDesc* desc = nullptr;
    if (desc_va != 0) {
        GuestShaderResourceViewDesc g{};
        if (!com::read_guest(desc_va, g)) {
            note_bad_argument("CreateShaderResourceView: unreadable descriptor");
            return;
        }
        if (!parse_srv_desc(g, &vd)) {
            note_bad_argument("CreateShaderResourceView: invalid view dimension " + std::to_string(g.ViewDimension));
            return;
        }
        desc = &vd;
    }
    if (va_watch::active()) {
        va_watch_resource("descriptor SRV", static_cast<const Resource*>(resource.get()), "CreateShaderResourceView at handle " + hex(dest), dest);
    }
    const d3d::Status s = heaps_.create_shader_resource_view(resource_id(resource), desc, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateShaderResourceView: ") + d3d::status_name(s));
        return;
    }
    store_descriptor_words(dest);
}

void Device::CreateUnorderedAccessView(InterfaceArg<IResourceHost> resource, InterfaceArg<IResourceHost> counter,
                                       uint64_t desc_va, uint64_t dest) {
    d3d::ViewDesc vd;
    const d3d::ViewDesc* desc = nullptr;
    if (desc_va != 0) {
        GuestUnorderedAccessViewDesc g{};
        if (!com::read_guest(desc_va, g)) {
            note_bad_argument("CreateUnorderedAccessView: unreadable descriptor");
            return;
        }
        if (!parse_uav_desc(g, &vd)) {
            note_bad_argument("CreateUnorderedAccessView: invalid view dimension " + std::to_string(g.ViewDimension));
            return;
        }
        desc = &vd;
    }
    if (va_watch::active()) {
        va_watch_resource("descriptor UAV", static_cast<const Resource*>(resource.get()), "CreateUnorderedAccessView at handle " + hex(dest), dest);
    }
    const d3d::Status s = heaps_.create_unordered_access_view(resource_id(resource), resource_id(counter), desc,
                                                              d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateUnorderedAccessView: ") + d3d::status_name(s));
        return;
    }
    store_descriptor_words(dest);
}

void Device::CreateRenderTargetView(InterfaceArg<IResourceHost> resource, uint64_t desc_va, uint64_t dest) {
    d3d::ViewDesc vd;
    const d3d::ViewDesc* desc = nullptr;
    if (desc_va != 0) {
        GuestRenderTargetViewDesc g{};
        if (!com::read_guest(desc_va, g)) {
            note_bad_argument("CreateRenderTargetView: unreadable descriptor");
            return;
        }
        vd.format = g.Format;
        vd.dimension = g.ViewDimension;
        const uint32_t* u = g.u;
        switch (g.ViewDimension) {
        case d3d::ViewDimension::Buffer:
            vd.first_element = u64_at(u, 0);
            vd.num_elements = u[2];
            break;
        case d3d::ViewDimension::Texture1D:
            vd.first_mip = u[0];
            break;
        case d3d::ViewDimension::Texture1DArray:
        case d3d::ViewDimension::Texture3D:
            vd.first_mip = u[0];
            vd.first_array_slice = u[1];
            vd.array_size = u[2];
            break;
        case d3d::ViewDimension::Texture2D:
            vd.first_mip = u[0];
            vd.plane_slice = u[1];
            break;
        case d3d::ViewDimension::Texture2DArray:
            vd.first_mip = u[0];
            vd.first_array_slice = u[1];
            vd.array_size = u[2];
            vd.plane_slice = u[3];
            break;
        case d3d::ViewDimension::Texture2DMs:
            break;
        case d3d::ViewDimension::Texture2DMsArray:
            vd.first_array_slice = u[0];
            vd.array_size = u[1];
            break;
        default:
            note_bad_argument("CreateRenderTargetView: invalid view dimension " + std::to_string(g.ViewDimension));
            return;
        }
        desc = &vd;
    }
    if (va_watch::active()) {
        va_watch_resource("descriptor RTV", static_cast<const Resource*>(resource.get()), "CreateRenderTargetView at handle " + hex(dest));
    }
    const d3d::Status s = heaps_.create_render_target_view(resource_id(resource), desc, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateRenderTargetView: ") + d3d::status_name(s));
    }
}

void Device::CreateDepthStencilView(InterfaceArg<IResourceHost> resource, uint64_t desc_va, uint64_t dest) {
    d3d::ViewDesc vd;
    const d3d::ViewDesc* desc = nullptr;
    if (desc_va != 0) {
        GuestDepthStencilViewDesc g{};
        if (!com::read_guest(desc_va, g)) {
            note_bad_argument("CreateDepthStencilView: unreadable descriptor");
            return;
        }
        vd.format = g.Format;
        vd.dimension = g.ViewDimension;
        vd.flags = g.Flags;
        const uint32_t* u = g.u;
        switch (g.ViewDimension) {
        case d3d::DsvDimension::Texture1D:
        case d3d::DsvDimension::Texture2D:
            vd.first_mip = u[0];
            break;
        case d3d::DsvDimension::Texture1DArray:
        case d3d::DsvDimension::Texture2DArray:
            vd.first_mip = u[0];
            vd.first_array_slice = u[1];
            vd.array_size = u[2];
            break;
        case d3d::DsvDimension::Texture2DMs:
            break;
        case d3d::DsvDimension::Texture2DMsArray:
            vd.first_array_slice = u[0];
            vd.array_size = u[1];
            break;
        default:
            note_bad_argument("CreateDepthStencilView: invalid view dimension " + std::to_string(g.ViewDimension));
            return;
        }
        desc = &vd;
    }
    const d3d::Status s = heaps_.create_depth_stencil_view(resource_id(resource), desc, d3d::CpuDescriptorHandle{dest});
    if (s != d3d::Status::Ok) {
        note_bad_argument(std::string("CreateDepthStencilView: ") + d3d::status_name(s));
    }
}

// ---------------------------------------------------------------------------------------------
// Pipeline states
// ---------------------------------------------------------------------------------------------

PipelineState::PipelineState(com::Ref<Device> device, com::Ref<RootSignature> root_signature)
    : ChildBase(std::move(device)), root_signature_(std::move(root_signature)) {
    device_->pipeline_states.fetch_add(1);
}

PipelineState::~PipelineState() {
    if (graphics_ || mesh_) {
        CommandEncoder enc;
        if (graphics_) {
            device_->state_bridge().release_pipeline_state(std::move(graphics_), enc);
        }
        if (mesh_) {
            device_->state_bridge().release_mesh_pipeline_state(std::move(mesh_), enc);
        }
        if (enc.size() != 0) {
            device_->submit(enc);
        }
    }
    device_->pipeline_states.fetch_sub(1);
}

HResult PipelineState::GetCachedBlob(OutPtr blob) {
    if (blob.is_null()) {
        return kEPointer;
    }
    auto b = Blob::create(nullptr, 0);
    blob.write(b ? b.detach(kID3DBlob) : 0);
    return b ? kSOk : kEOutOfMemory;
}

namespace {

// The shader kind a DXIL container declares (DXIL part, program header: kind = version >> 16:
// 0 pixel, 1 vertex, 2 geometry, 3 hull, 4 domain, 5 compute, 13 mesh, 14 amplification, ...),
// or -1 when the bytes are not a DXBC container with a DXIL part.
int dxil_shader_kind(const uint8_t* p, size_t n) {
    auto u32 = [&](size_t at) {
        uint32_t v = 0;
        std::memcpy(&v, p + at, 4);
        return v;
    };
    if (n < 32 || u32(0) != 0x43425844u) {  // "DXBC"
        return -1;
    }
    const uint32_t parts = u32(28);
    if (parts > 64 || 32 + uint64_t{parts} * 4 > n) {
        return -1;
    }
    for (uint32_t i = 0; i < parts; ++i) {
        const uint32_t off = u32(32 + i * 4);
        if (uint64_t{off} + 12 > n) {
            return -1;
        }
        const uint32_t fourcc = u32(off);
        if (fourcc == 0x4C495844u) {  // "DXIL"
            return static_cast<int>(u32(off + 8) >> 16);
        }
    }
    return -1;
}

// The 16-byte digest of a DXBC container's HASH part (after its u32 flags); false without one.
bool dxil_hash_digest(const uint8_t* p, size_t n, Device::DxilDigest* out) {
    auto u32 = [&](size_t at) {
        uint32_t v = 0;
        std::memcpy(&v, p + at, 4);
        return v;
    };
    if (n < 32 || u32(0) != 0x43425844u) {  // "DXBC"
        return false;
    }
    const uint32_t parts = u32(28);
    if (parts > 64 || 32 + uint64_t{parts} * 4 > n) {
        return false;
    }
    for (uint32_t i = 0; i < parts; ++i) {
        const uint32_t off = u32(32 + i * 4);
        if (uint64_t{off} + 8 > n) {
            return false;
        }
        if (u32(off) == 0x48534148u && u32(off + 4) >= 20 && uint64_t{off} + 8 + 20 <= n) {  // "HASH"
            std::memcpy(out->data(), p + off + 12, 16);
            return true;
        }
    }
    return false;
}

// The console-native program of `want` the container carries in its DXIL part
// (xbox_embedded_native.h), as a native stage record; written to the pipeline dump like a slot-57
// stage (metadata: the unpacked FSSM blob). Stays empty without one.
void embedded_native_stage(Device& dev, const std::vector<uint8_t>& bytes, xbox::EmbeddedStage want,
                           Device::NativeStageRecord* out) {
    *out = Device::NativeStageRecord{};
    xbox::EmbeddedNative native;
    std::string error;
    if (!xbox::extract_embedded_native(bytes.data(), bytes.size(), &native, &error)) {
        return;
    }
    const xbox::EmbeddedProgram* p = xbox::find_embedded(native, want);
    if (p == nullptr) {
        return;
    }
    out->code_identity = shader_identity(p->code.data(), p->code.size());
    out->identity = native_stage_identity(p->code.data(), p->code.size(), native.fssm.data(), native.fssm.size());
    out->code_bytes = static_cast<uint32_t>(p->code.size());
    out->metadata_bytes = static_cast<uint32_t>(native.fssm.size());
    if (PipelineDump* dump = dev.pipeline_dump()) {
        const auto kind = want == xbox::EmbeddedStage::Ps   ? PipelineDump::NativeStageKind::Pixel
                          : want == xbox::EmbeddedStage::Cs ? PipelineDump::NativeStageKind::Compute
                          : want == xbox::EmbeddedStage::Ms ? PipelineDump::NativeStageKind::Mesh
                                                            : PipelineDump::NativeStageKind::Vertex;
        dump->native_stage(out->code_identity, out->identity, kind, /*layout_b=*/false, p->code.data(), p->code.size(),
                           native.fssm.data(), native.fssm.size());
    }
}

// Reads a shader's bytecode and returns its identity (0 = no shader). False on unreadable input.
// `digest` (optional) receives the container's HASH digest; it stays all-zero without one.
// `embedded` (optional) receives the container's own native stage of `want` (embedded_native_stage).
bool read_shader(Device& dev, const GuestShaderBytecode& code, const char* stage, uint64_t* identity,
                 Device::DxilDigest* digest = nullptr, Device::NativeStageRecord* embedded = nullptr,
                 xbox::EmbeddedStage want = xbox::EmbeddedStage::Vs) {
    *identity = 0;
    if (digest != nullptr) {
        digest->fill(0);
    }
    if (embedded != nullptr) {
        *embedded = Device::NativeStageRecord{};
    }
    if (code.pShaderBytecode == 0 || code.BytecodeLength == 0) {
        return true;
    }
    if (code.BytecodeLength > kMaxShaderBytes) {
        dev.note_bad_argument(std::string("PSO: ") + stage + " bytecode too large");
        return false;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(code.BytecodeLength));
    if (!com::read_guest(code.pShaderBytecode, bytes.data(), bytes.size())) {
        dev.note_bad_argument(std::string("PSO: unreadable ") + stage + " bytecode at " + hex(code.pShaderBytecode));
        return false;
    }
    *identity = shader_identity(bytes.data(), bytes.size());
    dev.note_shader_size(*identity, bytes.size(), dxil_shader_kind(bytes.data(), bytes.size()));
    Device::DxilDigest d{};
    const bool has_digest = dxil_hash_digest(bytes.data(), bytes.size(), &d);
    if (digest != nullptr) {
        *digest = has_digest ? d : Device::DxilDigest{};
    }
    if (PipelineDump* dump = dev.pipeline_dump()) {
        dump->dxil(*identity, has_digest ? &d : nullptr, dxil_shader_kind(bytes.data(), bytes.size()), bytes.data(),
                   bytes.size());
    }
    if (embedded != nullptr) {
        const uint8_t stage_key = static_cast<uint8_t>(want);
        if (dev.pipeline_dump() == nullptr && dev.find_cached_embedded_native(*identity, stage_key, embedded)) {
            return true;
        }
        embedded_native_stage(dev, bytes, want, embedded);
        dev.cache_embedded_native(*identity, stage_key, *embedded);
    }
    return true;
}

com::Ref<RootSignature> root_signature_ref(uint64_t va) {
    if (va == 0) {
        return {};
    }
    auto arg = InterfaceArg<IRootSignatureHost>::resolve(va);
    return arg ? com::Ref<RootSignature>::share(static_cast<RootSignature*>(arg.get())) : com::Ref<RootSignature>{};
}

}  // namespace

GraphicsPsoSource& GraphicsPsoSource::operator=(const GraphicsPsoSource& other) {
    if (this != &other) {
        desc = other.desc;
        semantic_names = other.semantic_names;
        input = other.input;
        root_signature = other.root_signature;
        rect_list = other.rect_list;
        native_vs = other.native_vs;
        pack_vs_key = other.pack_vs_key;
        pack_ps_key = other.pack_ps_key;
        vs_fetches = other.vs_fetches;
        reflected = other.reflected;
        vs_digest = other.vs_digest;
        ps_digest = other.ps_digest;
        native_ps = other.native_ps;
        fix_pointers();
    }
    return *this;
}

void GraphicsPsoSource::fix_pointers() {
    for (size_t i = 0; i < input.size() && i < semantic_names.size(); ++i) {
        input[i].SemanticName = semantic_names[i].c_str();
    }
    desc.pInputElementDescs = input.empty() ? nullptr : input.data();
    desc.NumInputElements = static_cast<uint32_t>(input.size());
    desc.vs_input_fetches = reflected ? vs_fetches.data() : nullptr;
    desc.vs_input_fetch_count = reflected ? static_cast<uint32_t>(vs_fetches.size()) : 0u;
}

void convert_blend_desc(const GuestBlendDesc& g, pso::D3D12BlendDesc* out) {
    out->AlphaToCoverageEnable = g.AlphaToCoverageEnable != 0;
    out->IndependentBlendEnable = g.IndependentBlendEnable != 0;
    for (int i = 0; i < 8; ++i) {
        const GuestRenderTargetBlendDesc& s = g.RenderTarget[i];
        pso::D3D12RenderTargetBlendDesc& t = out->RenderTarget[i];
        t.BlendEnable = s.BlendEnable != 0;
        t.LogicOpEnable = s.LogicOpEnable != 0;
        t.SrcBlend = static_cast<pso::D3D12Blend>(s.SrcBlend);
        t.DestBlend = static_cast<pso::D3D12Blend>(s.DestBlend);
        t.BlendOp = static_cast<pso::D3D12BlendOp>(s.BlendOp);
        t.SrcBlendAlpha = static_cast<pso::D3D12Blend>(s.SrcBlendAlpha);
        t.DestBlendAlpha = static_cast<pso::D3D12Blend>(s.DestBlendAlpha);
        t.BlendOpAlpha = static_cast<pso::D3D12BlendOp>(s.BlendOpAlpha);
        t.LogicOp = static_cast<pso::D3D12LogicOp>(s.LogicOp);
        t.RenderTargetWriteMask = s.RenderTargetWriteMask;
    }
}

void convert_rasterizer_desc(const GuestRasterizerDesc& r, pso::D3D12RasterizerDesc* out) {
    out->FillMode = static_cast<pso::D3D12FillMode>(r.FillMode);
    out->CullMode = static_cast<pso::D3D12CullMode>(r.CullMode);
    out->FrontCounterClockwise = r.FrontCounterClockwise != 0;
    out->DepthBias = r.DepthBias;
    out->DepthBiasClamp = r.DepthBiasClamp;
    out->SlopeScaledDepthBias = r.SlopeScaledDepthBias;
    out->DepthClipEnable = r.DepthClipEnable != 0;
    out->MultisampleEnable = r.MultisampleEnable != 0;
    out->AntialiasedLineEnable = r.AntialiasedLineEnable != 0;
    out->ForcedSampleCount = r.ForcedSampleCount;
    out->ConservativeRaster = static_cast<pso::D3D12ConservativeRasterizationMode>(r.ConservativeRaster);
}

void convert_depth_stencil_desc(const GuestDepthStencilDesc& ds, pso::D3D12DepthStencilDesc* out) {
    out->DepthEnable = ds.DepthEnable != 0;
    out->DepthWriteMask = static_cast<pso::D3D12DepthWriteMask>(ds.DepthWriteMask);
    out->DepthFunc = static_cast<pso::D3D12ComparisonFunc>(ds.DepthFunc);
    out->StencilEnable = ds.StencilEnable != 0;
    out->StencilReadMask = ds.StencilReadMask;
    out->StencilWriteMask = ds.StencilWriteMask;
    auto op = [](const GuestDepthStencilOpDesc& s) {
        pso::D3D12DepthStencilOpDesc o;
        o.StencilFailOp = static_cast<pso::D3D12StencilOp>(s.StencilFailOp);
        o.StencilDepthFailOp = static_cast<pso::D3D12StencilOp>(s.StencilDepthFailOp);
        o.StencilPassOp = static_cast<pso::D3D12StencilOp>(s.StencilPassOp);
        o.StencilFunc = static_cast<pso::D3D12ComparisonFunc>(s.StencilFunc);
        return o;
    };
    out->FrontFace = op(ds.FrontFace);
    out->BackFace = op(ds.BackFace);
}

void PipelineState::initialize_guest_payload() {
    const uint64_t payload[2] = {xbox::kPipelineStateTag, identity()};
    com::write_guest(payload_address(), payload, sizeof(payload));
}

HResult Device::CreateGraphicsPipelineState(uint64_t desc_va, const Guid& riid, OutPtr out) {
    os::win32::HostStallScope stall;  // pipeline/shader builds can block the guest thread
    GuestGraphicsPipelineStateDesc g{};
    if (desc_va == 0 || !com::read_guest(desc_va, g)) {
        note_bad_argument("CreateGraphicsPipelineState: unreadable descriptor");
        return kEInvalidArg;
    }
    return create_graphics_pso(g, riid, out);
}

HResult Device::create_graphics_pso(const GuestGraphicsPipelineStateDesc& g, const Guid& riid, OutPtr out) {
    GraphicsPsoSource src;
    src.root_signature = root_signature_ref(g.pRootSignature);
    if (g.pRootSignature != 0 && !src.root_signature) {
        note_bad_argument("CreateGraphicsPipelineState: invalid root signature " + hex(g.pRootSignature));
        return kEInvalidArg;
    }
    pso::D3D12GraphicsPsoDesc& d = src.desc;
    NativeStageRecord embedded_vs, embedded_ps;
    if (!read_shader(*this, g.VS, "VS", &d.vs, &src.vs_digest, &embedded_vs, xbox::EmbeddedStage::Vs) ||
        !read_shader(*this, g.PS, "PS", &d.ps, &src.ps_digest, &embedded_ps, xbox::EmbeddedStage::Ps) ||
        !read_shader(*this, g.DS, "DS", &d.ds) || !read_shader(*this, g.HS, "HS", &d.hs) ||
        !read_shader(*this, g.GS, "GS", &d.gs)) {
        return kEInvalidArg;
    }
    if (d.vs == 0) {
        note_bad_argument("CreateGraphicsPipelineState: no vertex shader");
        return kEInvalidArg;
    }
    // Xbox rectangle topology type: kept as triangles plus a flag (d3d12_xs_xbox.h).
    src.rect_list = g.PrimitiveTopologyType == xbox::kTopologyTypeRect;
    const uint32_t topology_type = src.rect_list ? 3u : g.PrimitiveTopologyType;
    if (g.NumRenderTargets > 8 || g.InputLayout.NumElements > kMaxInputElements || topology_type > 4) {
        note_bad_argument("CreateGraphicsPipelineState: out-of-range counts or topology type (render targets " +
                          std::to_string(g.NumRenderTargets) + ", input elements " +
                          std::to_string(g.InputLayout.NumElements) + ", topology type " +
                          std::to_string(g.PrimitiveTopologyType) + ")");
        return kEInvalidArg;
    }
    if (src.rect_list) {
        count(xbox::Counter::RectListPipelineStates);
    }
    // Join each DXIL stage to the console-native stage compiled from the same shader (only stages
    // indexed by slot 57 with refuse_native_pipelines; d3d12_xs_xbox.h).
    if (native_index_size() != 0) {
        static const DxilDigest kZero{};
        for (auto [digest, record] : {std::pair{&src.vs_digest, &src.native_vs}, std::pair{&src.ps_digest, &src.native_ps}}) {
            if (*digest == kZero) {
                continue;
            }
            count(find_native_stage(*digest, record) ? xbox::Counter::DxilStagesJoined
                                                     : xbox::Counter::DxilStagesNotJoined);
        }
    }
    // A stage no slot-57 stage joins runs the native code its own container carries (G66).
    for (auto [embedded, record] : {std::pair{&embedded_vs, &src.native_vs}, std::pair{&embedded_ps, &src.native_ps}}) {
        if (record->identity == 0 && embedded->identity != 0) {
            *record = *embedded;
            count(xbox::Counter::DxilStagesEmbeddedNative);
        }
    }
    d.stream_output_enabled = g.StreamOutput.NumEntries > 0;
    convert_blend_desc(g.BlendState, &d.BlendState);
    d.SampleMask = g.SampleMask;
    convert_rasterizer_desc(g.RasterizerState, &d.RasterizerState);
    convert_depth_stencil_desc(g.DepthStencilState, &d.DepthStencilState);
    // Input layout: semantic names are guest strings, copied into the source.
    std::vector<GuestInputElementDesc> elements(g.InputLayout.NumElements);
    src.semantic_names.assign(g.InputLayout.NumElements, std::string());
    src.input.assign(g.InputLayout.NumElements, pso::D3D12InputElementDesc{});
    for (uint32_t i = 0; i < g.InputLayout.NumElements; ++i) {
        if (!com::read_guest(g.InputLayout.pInputElementDescs + sizeof(GuestInputElementDesc) * i, elements[i])) {
            note_bad_argument("CreateGraphicsPipelineState: unreadable input element " + std::to_string(i));
            return kEInvalidArg;
        }
        for (uint32_t c = 0; c < 64; ++c) {
            char ch = 0;
            if (!com::read_guest(elements[i].SemanticName + c, &ch, 1)) {
                note_bad_argument("CreateGraphicsPipelineState: unreadable semantic name of element " + std::to_string(i));
                return kEInvalidArg;
            }
            if (ch == 0) {
                break;
            }
            src.semantic_names[i].push_back(ch);
        }
    }
    for (uint32_t i = 0; i < g.InputLayout.NumElements; ++i) {
        const GuestInputElementDesc& e = elements[i];
        pso::D3D12InputElementDesc& t = src.input[i];
        t.SemanticIndex = e.SemanticIndex;
        t.Format = static_cast<pso::DxgiFormat>(e.Format);
        t.InputSlot = e.InputSlot;
        t.AlignedByteOffset = e.AlignedByteOffset;
        t.InputSlotClass = static_cast<pso::D3D12InputClassification>(e.InputSlotClass);
        t.InstanceDataStepRate = e.InstanceDataStepRate;
    }
    d.IBStripCutValue = static_cast<pso::D3D12IndexBufferStripCutValue>(g.IBStripCutValue);
    d.PrimitiveTopologyType = static_cast<pso::D3D12PrimitiveTopologyType>(topology_type);
    d.NumRenderTargets = g.NumRenderTargets;
    for (int i = 0; i < 8; ++i) {
        d.RTVFormats[i] = static_cast<pso::DxgiFormat>(g.RTVFormats[i]);
    }
    d.DSVFormat = static_cast<pso::DxgiFormat>(g.DSVFormat);
    d.SampleDesc.Count = g.SampleCount;
    d.SampleDesc.Quality = g.SampleQuality;
    return create_graphics_pso_from(std::move(src), riid, out);
}

// Pipeline layout from a root signature (convention in Device::resolve_with_pack).
bridge::PipelineLayoutDesc layout_from_root_signature(const d3d::RootSignatureDesc& rs) {
    // WebGPU core per-stage binding limits per resource class: keep every resource class
    // represented and every emitted BindGroupLayout valid even when a root signature declares
    // oversized descriptor tables (e.g. 240 SRVs / 19 UAVs / 17 CBVs / 24 static samplers).
    constexpr uint32_t kMaxCbvBindingsPerStage = 12u;
    constexpr uint32_t kMaxSrvBindingsPerStage = 16u;
    constexpr uint32_t kMaxUavBindingsPerStage = 8u;
    constexpr uint32_t kMaxSamplerBindingsPerStage = 16u;

    bridge::GroupLayoutDesc group;
    uint32_t binding = 0;
    uint32_t cbv_count = 0;
    uint32_t srv_count = 0;
    uint32_t uav_count = 0;
    uint32_t sampler_count = 0;
    auto add = [&](GpuBindingLayoutType type, uint8_t binding_type, uint8_t view_dimension, d3d::DescriptorRangeType cls,
                   uint32_t reg, uint32_t space, bool root_constants) {
        bridge::LayoutBinding b;
        b.entry.binding = binding;
        b.entry.visibility = GpuShaderStageBits::Vertex | GpuShaderStageBits::Fragment;
        b.entry.type = static_cast<uint8_t>(type);
        b.entry.binding_type = binding_type;
        b.entry.view_dimension = view_dimension;
        b.join.binding = binding;
        b.join.register_class = cls;
        b.join.shader_register = reg;
        b.join.register_space = space;
        b.join.root_constants = root_constants;
        group.bindings.push_back(b);
        ++binding;
    };
    auto add_class = [&](d3d::DescriptorRangeType cls, uint32_t reg, uint32_t space, bool root_constants) {
        switch (cls) {
            case d3d::DescriptorRangeType::Cbv:
                if (cbv_count >= kMaxCbvBindingsPerStage) {
                    return;
                }
                ++cbv_count;
                add(GpuBindingLayoutType::Buffer, static_cast<uint8_t>(GpuBufferBindingType::Uniform), 0, cls, reg, space,
                    root_constants);
                break;
            case d3d::DescriptorRangeType::Srv:
                if (srv_count >= kMaxSrvBindingsPerStage) {
                    return;
                }
                ++srv_count;
                add(GpuBindingLayoutType::Texture, static_cast<uint8_t>(GpuTextureSampleType::Float),
                    static_cast<uint8_t>(GpuTextureViewDimension::Dim2D), cls, reg, space, false);
                break;
            case d3d::DescriptorRangeType::Uav:
                if (uav_count >= kMaxUavBindingsPerStage) {
                    return;
                }
                ++uav_count;
                add(GpuBindingLayoutType::Buffer, static_cast<uint8_t>(GpuBufferBindingType::Storage), 0, cls, reg, space,
                    false);
                break;
            case d3d::DescriptorRangeType::Sampler:
                if (sampler_count >= kMaxSamplerBindingsPerStage) {
                    return;
                }
                ++sampler_count;
                add(GpuBindingLayoutType::Sampler, static_cast<uint8_t>(GpuSamplerBindingType::Filtering), 0, cls, reg,
                    space, false);
                break;
        }
    };
    for (const d3d::RootParameterDesc& p : rs.parameters) {
        switch (p.parameter_type) {
            case d3d::RootParameterType::DescriptorTable:
                for (const d3d::DescriptorRangeDesc& r : p.ranges) {
                    const uint32_t n = r.num_descriptors == d3d::kUnboundedDescriptorCount ? 1 : r.num_descriptors;
                    for (uint32_t i = 0; i < n; ++i) {
                        add_class(r.range_type, r.base_shader_register + i, r.register_space, false);
                    }
                }
                break;
            case d3d::RootParameterType::Constants32Bit:
                add_class(d3d::DescriptorRangeType::Cbv, p.shader_register, p.register_space, true);
                break;
            case d3d::RootParameterType::Cbv:
                add_class(d3d::DescriptorRangeType::Cbv, p.shader_register, p.register_space, false);
                break;
            case d3d::RootParameterType::Srv:
                add_class(d3d::DescriptorRangeType::Srv, p.shader_register, p.register_space, false);
                break;
            case d3d::RootParameterType::Uav:
                add_class(d3d::DescriptorRangeType::Uav, p.shader_register, p.register_space, false);
                break;
        }
    }
    for (const d3d::StaticSamplerDesc& s : rs.static_samplers) {
        add_class(d3d::DescriptorRangeType::Sampler, s.shader_register, s.register_space, false);
    }
    bridge::PipelineLayoutDesc layout;
    layout.groups.push_back(std::move(group));
    return layout;
}

size_t Device::reflection_join_failures() const {
    std::lock_guard<std::mutex> lock(pack_mutex_);
    return reflection_failures_.size();
}

uint32_t Device::module_locked(uint64_t key, GpuShaderStage stage, const std::string& code) {
    if (options_.shader_pack && !pack_table_) {
        pack_table_ = std::make_unique<PackModuleTable>(*options_.shader_pack);
    }
    if (pack_table_) {
        const uint32_t id = pack_table_->get_or_create(key, [&](uint32_t new_id) {
            CommandEncoder enc;
            enc.create_shader_module(new_id, stage, code);
            submit(enc);
        });
        if (id != 0) {
            return id;
        }
    }
    auto it = pack_modules_.find(key);
    if (it != pack_modules_.end()) {
        return it->second;
    }
    const uint32_t id = allocate_id();
    CommandEncoder enc;
    enc.create_shader_module(id, stage, code);
    submit(enc);
    pack_modules_.emplace(key, id);
    return id;
}

uint32_t Device::pack_binding_shift(uint64_t key) const {
    const StageReflectionIndex* refl = options_.stage_reflection.get();
    if (refl == nullptr) {
        return 0;
    }
    const auto it = refl->find(key);
    return it != refl->end() ? it->second.group0_binding_shift : 0;
}

uint32_t Device::pack_module_locked(uint64_t key, GpuShaderStage stage) {
    return module_locked(key, stage, pack_key_code(key, pack_binding_shift(key)));
}

bool Device::prewarm_pack_modules(uint32_t kind, uint64_t key0, uint64_t key1) {
    if (!options_.shader_pack || key0 == 0) {
        return false;
    }
    const ShaderPackIndex& pack = *options_.shader_pack;
    if (pack.find(key0) == pack.end() || (key1 != 0 && pack.find(key1) == pack.end())) {
        return false;
    }
    std::lock_guard<std::mutex> lock(pack_mutex_);
    if (kind == kPrewarmKindCompute) {
        pack_module_locked(key0, GpuShaderStage::Compute);
        return true;
    }
    pack_module_locked(key0, GpuShaderStage::Vertex);
    if (key1 != 0) {
        pack_module_locked(key1, GpuShaderStage::Fragment);
    }
    return true;
}

bridge::PrewarmPipelineResult Device::prewarm_pipeline(uint32_t task_id, uint32_t kind, uint64_t key0, uint64_t key1,
                                                       const std::vector<uint8_t>& detail) {
    using Result = bridge::PrewarmPipelineResult;
    PrewarmPipelineDesc desc;
    if (!options_.shader_pack || !decode_prewarm_pipeline_desc(detail.data(), detail.size(), &desc) ||
        desc.kind != kind || desc.pack_key0 != key0 || desc.pack_key1 != key1) {
        return Result::Failed;
    }
    if (kind != kPrewarmKindCompute && key1 != 0 && pack_binding_shift(key1) != 0 && desc.has_layout) {
        // A warm list recorded before the fragment stage was renumbered holds a layout for the old
        // numbers: leave that pipeline to the title's own (reflected) request.
        const StageReflection& ps = options_.stage_reflection->at(key1);
        for (const ReflectedBinding& b : ps.bindings) {
            if (b.group != 0) {
                continue;
            }
            const bool listed =
                !desc.layout.groups.empty() &&
                std::any_of(desc.layout.groups[0].bindings.begin(), desc.layout.groups[0].bindings.end(),
                            [&](const bridge::LayoutBinding& lb) { return lb.entry.binding == b.binding; });
            if (!listed) {
                return Result::Failed;
            }
        }
    }
    if (!prewarm_pack_modules(kind, key0, key1)) {
        return Result::Failed;
    }
    uint32_t vs_module = 0;
    uint32_t fs_module = 0;
    {
        std::lock_guard<std::mutex> lock(pack_mutex_);
        if (pack_table_) {
            vs_module = pack_table_->id_of(key0);
            fs_module = key1 != 0 ? pack_table_->id_of(key1) : 0;
        }
    }
    return bridge_->prewarm_pipeline(desc, vs_module, fs_module, task_id, [this](CommandEncoder& enc) { submit(enc); });
}

bool Device::resolve_with_pack(GraphicsPsoSource& src, uint64_t* vs_key, uint64_t* ps_key) {
    *vs_key = *ps_key = 0;
    src.reflected = false;
    src.vs_fetches.clear();
    src.fix_pointers();
    const ShaderPackIndex* index = options_.shader_pack.get();
    const pso::D3D12GraphicsPsoDesc& d = src.desc;
    if (index == nullptr || d.vs == 0 || d.ds != 0 || d.hs != 0 || d.gs != 0) {
        return false;
    }
    auto key_for = [&](uint64_t dxil, const NativeStageRecord& native, PackStage stage, uint64_t* key) {
        for (uint64_t k : {native.identity != 0 ? native.code_identity : 0, dxil}) {
            auto it = k != 0 ? index->find(k) : index->end();
            if (it != index->end() && it->second == stage) {
                *key = k;
                return true;
            }
        }
        return false;
    };
    if (!key_for(d.vs, src.native_vs, PackStage::Vertex, vs_key) ||
        (d.ps != 0 && !key_for(d.ps, src.native_ps, PackStage::Fragment, ps_key))) {
        *vs_key = *ps_key = 0;
        return false;
    }
    // Reflection join (G62): when every stage has reflection, the vertex attributes and the
    // pipeline layout come from it; a pipeline that cannot be joined is not resolved.
    const StageReflectionIndex* refl = options_.stage_reflection.get();
    const StageReflection* vs_refl = nullptr;
    const StageReflection* ps_refl = nullptr;
    if (refl != nullptr) {
        auto vit = refl->find(*vs_key);
        auto pit = d.ps != 0 ? refl->find(*ps_key) : refl->end();
        vs_refl = vit != refl->end() ? &vit->second : nullptr;
        ps_refl = pit != refl->end() ? &pit->second : nullptr;
        if (d.ps != 0 && (vs_refl == nullptr) != (ps_refl == nullptr)) {
            vs_refl = ps_refl = nullptr;  // only one stage reflected: keep the conventions for both
        }
    }
    bridge::PipelineLayoutDesc reflected_layout;
    if (vs_refl != nullptr) {
        std::string report;
        bool ok = static_cast<bool>(src.root_signature);
        if (!ok) {
            report = "no root signature\n";
        }
        std::vector<pso::VertexInputFetch> fetches;
        if (ok) {
            JoinStats join_stats;
            ok = join_vertex_inputs(*vs_refl, src.input.data(), static_cast<uint32_t>(src.input.size()), &fetches,
                                    &report, &join_stats);
            std::vector<ReflectedStageRef> stages{{GpuShaderStageBits::Vertex, vs_refl}};
            if (ps_refl != nullptr) {
                stages.push_back({GpuShaderStageBits::Fragment, ps_refl});
            }
            ok = join_pipeline_layout(src.root_signature->signature().desc(), /*compute=*/false, stages,
                                      &reflected_layout, &report, &join_stats,
                                      JoinOptions{options_.bridge.binding_arrays}) && ok;
            count_work(Work::ReflectionLaneIndexedUnbound, join_stats.lane_indexed_bindings);
            if (ok) {
                count_work(Work::ReflectionTableSlotsOutsideRs, join_stats.table_slots_outside_rs);
                count_work(Work::ReflectionRootDescriptorTables, join_stats.root_descriptor_tables);
                count_work(Work::ReflectionVertexInputsDefaulted, join_stats.vertex_inputs_defaulted);
                count_work(Work::ReflectionUnresolvedBindings, join_stats.unresolved_bindings);
            }
        }
        if (!ok) {
            const uint64_t rs_hash = src.root_signature ? src.root_signature->signature().hash() : 0;
            std::lock_guard<std::mutex> lock(pack_mutex_);
            if (reflection_failures_.insert({*vs_key, *ps_key, rs_hash}).second) {
                char head[96];
                std::snprintf(head, sizeof(head), "d3d12_xs: reflection join failed (vertex 0x%016" PRIx64
                              ", pixel 0x%016" PRIx64 "):\n", *vs_key, *ps_key);
                os::log(std::string(head) + report);
            }
            *vs_key = *ps_key = 0;
            return false;
        }
        src.vs_fetches = std::move(fetches);
        src.reflected = true;
        src.fix_pointers();
    }
    std::lock_guard<std::mutex> lock(pack_mutex_);
    auto module_for = [&](uint64_t key, GpuShaderStage stage) { return pack_module_locked(key, stage); };
    pipeline_prewarm().note_title_pipeline(kPrewarmKindGraphics, *vs_key, d.ps != 0 ? *ps_key : 0);
    register_shader_module(d.vs, module_for(*vs_key, GpuShaderStage::Vertex));
    if (d.ps != 0) {
        register_shader_module(d.ps, module_for(*ps_key, GpuShaderStage::Fragment));
    }
    if (src.root_signature && pack_layouts_.insert({d.vs, d.ps}).second) {
        bridge_->register_pipeline_layout(d.vs, d.ps,
                                          src.reflected ? std::move(reflected_layout)
                                                        : layout_from_root_signature(src.root_signature->signature().desc()));
    }
    return true;
}

HResult Device::create_graphics_pso_from(GraphicsPsoSource source, const Guid& riid, OutPtr out) {
    auto src = std::make_unique<GraphicsPsoSource>(source);  // copy: fixes the internal pointers
    const com::Ref<RootSignature>& rs = src->root_signature;
    pso::D3D12GraphicsPsoDesc& d = src->desc;
    d.root_signature_id = rs ? static_cast<uint32_t>(rs->signature().hash()) : 0u;
    resolve_with_pack(*src, &src->pack_vs_key, &src->pack_ps_key);
    auto state = com::create_object<PipelineState>(com::implements(kID3D12PipelineState), device_ref(this), rs);
    if (!state) {
        return kEOutOfMemory;
    }
    state->initialize_guest_payload();
    state->graphics_ = bridge_->create_graphics_pipeline_state(d, rs ? &rs->signature() : nullptr);
    state->identities_ = {d.vs, d.ps};
    state->resolved_ = (state->graphics_->diagnostics() & bridge::BridgeDiag::UnknownShaderModule) == 0;
    if (!state->resolved_) {
        if (d.vs != 0 && !shader_known(d.vs)) {
            note_unresolved_shader(d.vs, ShaderStage::Vertex);
        }
        if (d.ps != 0 && !shader_known(d.ps)) {
            note_unresolved_shader(d.ps, ShaderStage::Pixel);
        }
    }
    if (dump_) {
        state->dump_index_ = dump_->pipeline(*src, src->vs_digest, src->ps_digest, rs.get(),
                                             rs ? &rs->signature().desc() : nullptr);
    }
    state->source_ = std::move(src);
    return hand_out(std::move(state), riid, out);
}

bool Device::mesh_pipelines_possible() const {
    std::call_once(mesh_possible_once_, [this] {
        const ShaderPackIndex* pack = options_.shader_pack.get();
        const StageReflectionIndex* refl = options_.stage_reflection.get();
        if (pack == nullptr || refl == nullptr) {
            return;
        }
        for (const auto& [key, r] : *refl) {
            if (!r.has_mesh_output) {
                continue;
            }
            const auto it = pack->find(key);
            if (it != pack->end() && it->second == PackStage::Compute) {
                mesh_possible_ = true;
                return;
            }
        }
    });
    return mesh_possible_;
}

HResult Device::create_mesh_pso(const GuestGraphicsPipelineStateDesc& g, const GuestShaderBytecode& ms,
                                bool amplification, const Guid& riid, OutPtr out) {
    const auto placeholder = [&](const char* reason) {
        count(xbox::Counter::PlaceholderPipelineStates);
        return create_placeholder_pso(reason, riid, out);
    };
    if (amplification) {
        return placeholder("amplification shaders");
    }
    const ShaderPackIndex* pack = options_.shader_pack.get();
    const StageReflectionIndex* refl = options_.stage_reflection.get();
    com::Ref<RootSignature> rs = root_signature_ref(g.pRootSignature);
    if (pack == nullptr || refl == nullptr || !rs || g.NumRenderTargets > 8) {
        return placeholder("mesh shaders (no shader pack, reflection or root signature)");
    }
    // Fast gate: when the shader pack + reflection sidecar contain zero mesh-compute stages,
    // return a placeholder immediately before reading, hashing, or inflating MS/PS bytecodes
    // (avoids multi-second CPU stalls on 512+ mesh PSO waves during boot).
    if (dump_ == nullptr && ms.pShaderBytecode != 0 && ms.BytecodeLength != 0 &&
        ms.BytecodeLength <= kMaxShaderBytes && !mesh_pipelines_possible()) {
        return placeholder("mesh shaders (no mesh compute translation in the pack)");
    }
    uint64_t ms_id = 0;
    uint64_t ps_id = 0;
    DxilDigest ms_digest{};
    DxilDigest ps_digest{};
    // A stage no slot-57 stage joins runs the native code its own container carries, as for the
    // other stage kinds; the mesh stage's is also written to the pipeline dump (stage "mesh").
    NativeStageRecord embedded_ms{};
    NativeStageRecord embedded_ps{};
    if (!read_shader(*this, ms, "MS", &ms_id, &ms_digest, &embedded_ms, xbox::EmbeddedStage::Ms) || ms_id == 0) {
        return placeholder("mesh shaders (unreadable stage)");
    }
    // Pack keys, as for the other stages: the joined console-native stage's code identity first,
    // then the DXIL identity.
    static const DxilDigest kZero{};
    const auto find_key = [&](const DxilDigest& digest, uint64_t dxil, uint64_t salt, PackStage stage,
                              const NativeStageRecord& embedded) -> uint64_t {
        NativeStageRecord native{};
        if (native_index_size() != 0 && digest != kZero) {
            find_native_stage(digest, &native);
        }
        // The joined slot-57 stage first, then the container's own native stage, then the DXIL identity.
        for (uint64_t k : {native.identity != 0 ? native.code_identity : uint64_t{0},
                           embedded.identity != 0 ? embedded.code_identity : uint64_t{0}, dxil}) {
            auto it = k != 0 ? pack->find(k ^ salt) : pack->end();
            if (it != pack->end() && it->second == stage) {
                return k ^ salt;
            }
        }
        return 0;
    };
    // Resolve MS first so an MS pack/reflection miss skips reading/inflating the PS container.
    const uint64_t ms_key = find_key(ms_digest, ms_id, bridge::kMeshComputeKeySalt, PackStage::Compute, embedded_ms);
    const auto ms_refl = ms_key != 0 ? refl->find(ms_key) : refl->end();
    if (dump_ == nullptr && (ms_key == 0 || ms_refl == refl->end() || !ms_refl->second.has_mesh_output)) {
        return placeholder(ms_key == 0                        ? "mesh shaders (no mesh compute translation in the pack)"
                           : ms_refl == refl->end()           ? "mesh shaders (no reflection for the mesh compute stage)"
                                                              : "mesh shaders (reflection without mesh_output)");
    }
    if (!read_shader(*this, g.PS, "PS", &ps_id, &ps_digest, &embedded_ps, xbox::EmbeddedStage::Ps)) {
        return placeholder("mesh shaders (unreadable stage)");
    }
    const uint64_t ps_key = ps_id != 0 ? find_key(ps_digest, ps_id, 0, PackStage::Fragment, embedded_ps) : 0;
    if (ms_key == 0 || ms_refl == refl->end() || !ms_refl->second.has_mesh_output || (ps_id != 0 && ps_key == 0)) {
        // The reason names the missing piece (the text is the deduplication key of the notice).
        return placeholder(ms_key == 0                         ? "mesh shaders (no mesh compute translation in the pack)"
                           : ms_refl == refl->end()            ? "mesh shaders (no reflection for the mesh compute stage)"
                           : !ms_refl->second.has_mesh_output  ? "mesh shaders (reflection without mesh_output)"
                                                               : "mesh shaders (no pixel stage translation in the pack)");
    }
    const bridge::MeshOutputLayout output = ms_refl->second.mesh_output;
    // Layouts: the mesh stage's from its reflection (graphics user data, compute visibility); the
    // draw's from the pixel stage's reflection, or the root-signature convention without one.
    std::string report;
    bridge::PipelineLayoutDesc stage_layout;
    bool ok = join_pipeline_layout(rs->signature().desc(), /*compute=*/false,
                                   {{GpuShaderStageBits::Compute, &ms_refl->second}}, &stage_layout, &report, nullptr,
                                   JoinOptions{options_.bridge.binding_arrays, kMeshUserDataBase});
    bridge::PipelineLayoutDesc draw_layout;
    const auto ps_refl = ps_key != 0 ? refl->find(ps_key) : refl->end();
    if (ok && ps_refl != refl->end()) {
        ok = join_pipeline_layout(rs->signature().desc(), /*compute=*/false,
                                  {{GpuShaderStageBits::Fragment, &ps_refl->second}}, &draw_layout, &report, nullptr,
                                  JoinOptions{options_.bridge.binding_arrays});
    } else if (ok) {
        draw_layout = layout_from_root_signature(rs->signature().desc());
    }
    if (!ok || stage_layout.groups.size() > bridge::kMeshOutputGroup ||
        draw_layout.groups.size() > bridge::kMeshOutputGroup) {
        char head[96];
        std::snprintf(head, sizeof(head), "d3d12_xs: mesh pipeline layout join failed (mesh 0x%016" PRIx64 "):\n", ms_key);
        os::log(std::string(head) + report);
        return placeholder("mesh shaders (pipeline layout)");
    }
    const uint64_t stage_identity = bridge::mesh_compute_pack_key(ms_id);
    const uint64_t draw_vs = bridge::mesh_draw_vertex_identity(output);
    {
        std::lock_guard<std::mutex> lock(pack_mutex_);
        const auto module_for = [&](uint64_t key, GpuShaderStage stage, const std::string& code) {
            return module_locked(key, stage, code);
        };
        register_shader_module(stage_identity, module_for(ms_key, GpuShaderStage::Compute, pack_key_code(ms_key)));
        register_shader_module(draw_vs, module_for(draw_vs, GpuShaderStage::Vertex, bridge::mesh_draw_vertex_wgsl(output)));
        if (ps_id != 0) {
            register_shader_module(ps_id, module_for(ps_key, GpuShaderStage::Fragment,
                                                     pack_key_code(ps_key, pack_binding_shift(ps_key))));
        }
        if (pack_layouts_.insert({stage_identity, 0}).second) {
            bridge_->register_pipeline_layout(stage_identity, 0, std::move(stage_layout));
        }
        if (pack_layouts_.insert({draw_vs, ps_id}).second) {
            bridge_->register_pipeline_layout(draw_vs, ps_id, std::move(draw_layout));
        }
    }
    pso::D3D12GraphicsPsoDesc d{};
    d.vs = draw_vs;
    d.ps = ps_id;
    d.root_signature_id = static_cast<uint32_t>(rs->signature().hash());
    convert_blend_desc(g.BlendState, &d.BlendState);
    d.SampleMask = g.SampleMask;
    convert_rasterizer_desc(g.RasterizerState, &d.RasterizerState);
    convert_depth_stencil_desc(g.DepthStencilState, &d.DepthStencilState);
    d.PrimitiveTopologyType = pso::D3D12PrimitiveTopologyType::Triangle;
    d.NumRenderTargets = g.NumRenderTargets;
    for (int i = 0; i < 8; ++i) {
        d.RTVFormats[i] = static_cast<pso::DxgiFormat>(g.RTVFormats[i]);
    }
    d.DSVFormat = static_cast<pso::DxgiFormat>(g.DSVFormat);
    d.SampleDesc.Count = g.SampleCount;
    d.SampleDesc.Quality = g.SampleQuality;
    auto state = com::create_object<PipelineState>(com::implements(kID3D12PipelineState), device_ref(this), rs);
    if (!state) {
        return kEOutOfMemory;
    }
    state->initialize_guest_payload();
    state->mesh_ = bridge_->create_mesh_pipeline_state(d, stage_identity, output, &rs->signature());
    state->identities_ = {ms_id, ps_id};
    state->resolved_ = (state->mesh_->diagnostics() &
                        (bridge::BridgeDiag::UnknownShaderModule | bridge::BridgeDiag::MissingPipelineLayout)) == 0;
    count(xbox::Counter::MeshPipelineStates);
    return hand_out(std::move(state), riid, out);
}

HResult Device::create_placeholder_pso(const std::string& reason, const Guid& riid, OutPtr out) {
    auto state = com::create_object<PipelineState>(com::implements(kID3D12PipelineState), device_ref(this),
                                                   com::Ref<RootSignature>{});
    if (!state) {
        return kEOutOfMemory;
    }
    state->initialize_guest_payload();
    state->unsupported_ = reason;
    note_bad_argument("pipeline state not supported (" + reason + "): draws with it are skipped");
    return hand_out(std::move(state), riid, out);
}

bool Device::read_native_stage(const xbox::NativeStage& s, bool index, NativeStageRecord* out, bool pixel,
                               bool layout_b) {
    *out = NativeStageRecord{};
    if (!s.present()) {
        return true;
    }
    std::vector<uint8_t> code(static_cast<size_t>(s.code.size));
    std::vector<uint8_t> meta(static_cast<size_t>(s.metadata.size));
    if (!com::read_guest(s.code.address, code.data(), code.size()) ||
        !com::read_guest(s.metadata.address, meta.data(), meta.size())) {
        note_bad_argument("CreatePipelineState (native): unreadable stage at " + hex(s.code.address));
        return false;
    }
    out->identity = native_stage_identity(code.data(), code.size(), meta.data(), meta.size());
    out->code_identity = shader_identity(code.data(), code.size());
    out->code_bytes = static_cast<uint32_t>(code.size());
    out->metadata_bytes = static_cast<uint32_t>(meta.size());
    note_shader_size(out->identity, code.size());
    if (dump_) {
        dump_->native_stage(out->code_identity, out->identity, pixel, layout_b, code.data(), code.size(), meta.data(),
                            meta.size());
    }
    if (!index) {
        return true;
    }
    std::lock_guard<std::mutex> lock(shader_mutex_);
    if (!indexed_native_.insert(out->identity).second) {
        return true;
    }
    static const DxilDigest kZero{};
    for (size_t o = xbox::kNativeDigestFirst; o + 16 <= meta.size(); o += xbox::kNativeDigestStride) {
        DxilDigest d;
        std::memcpy(d.data(), meta.data() + o, 16);
        if (d != kZero) {
            native_by_digest_.emplace(d, *out);
        }
    }
    if (meta.size() >= 28) {
        DxilDigest d;
        std::memcpy(d.data(), meta.data() + meta.size() - 28, 16);
        if (d != kZero) {
            native_by_digest_.emplace(d, *out);
        }
    }
    return true;
}

bool Device::find_native_stage(const DxilDigest& digest, NativeStageRecord* out) const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    auto it = native_by_digest_.find(digest);
    if (it == native_by_digest_.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

size_t Device::native_index_size() const {
    std::lock_guard<std::mutex> lock(shader_mutex_);
    return indexed_native_.size();
}

bool Device::find_cached_embedded_native(uint64_t identity, uint8_t want, NativeStageRecord* out) const {
    if (identity == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(shader_mutex_);
    const auto it = embedded_native_cache_.find(EmbeddedCacheKey{identity, want});
    if (it == embedded_native_cache_.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

void Device::cache_embedded_native(uint64_t identity, uint8_t want, const NativeStageRecord& record) {
    if (identity == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(shader_mutex_);
    if (embedded_native_cache_.size() < kMaxEmbeddedNativeCache) {
        embedded_native_cache_.emplace(EmbeddedCacheKey{identity, want}, record);
    }
}

HResult Device::create_native_pso(const xbox::NativePipeline& pipeline, const Guid& riid, OutPtr out) {
    com::Ref<RootSignature> rs = root_signature_ref(pipeline.root_signature);
    if (pipeline.root_signature != 0 && !rs) {
        note_bad_argument("CreatePipelineState (native): invalid root signature " + hex(pipeline.root_signature));
        return kEInvalidArg;
    }
    auto native = std::make_unique<PipelineState::Native>();
    native->layout = pipeline;
    native->header.resize(static_cast<size_t>(pipeline.header.size));
    if (!com::read_guest(pipeline.header.address, native->header.data(), native->header.size())) {
        note_bad_argument("CreatePipelineState (native): unreadable header record at " + hex(pipeline.header.address));
        return kEInvalidArg;
    }
    NativeStageRecord vs, ps;
    if (!read_native_stage(pipeline.vertex, false, &vs, false, pipeline.vertex_alt) ||
        !read_native_stage(pipeline.pixel, false, &ps, true, pipeline.vertex_alt)) {
        return kEInvalidArg;
    }
    native->vertex_identity = vs.identity;
    native->pixel_identity = ps.identity;
    native->vertex_code_identity = vs.code_identity;
    native->pixel_code_identity = ps.code_identity;
    auto state = com::create_object<PipelineState>(com::implements(kID3D12PipelineState), device_ref(this), rs);
    if (!state) {
        return kEOutOfMemory;
    }
    state->initialize_guest_payload();
    state->identities_ = {native->vertex_identity, native->pixel_identity};
    state->unsupported_ = "console-native pipeline: GPU machine code and a register-state record, not translated";
    note_native_stages(native->vertex_identity, native->pixel_identity);
    state->native_ = std::move(native);
    return hand_out(std::move(state), riid, out);
}

HResult Device::CreateComputePipelineState(uint64_t desc_va, const Guid& riid, OutPtr out) {
    os::win32::HostStallScope stall;  // pipeline/shader builds can block the guest thread
    GuestComputePipelineStateDesc g{};
    if (desc_va == 0 || !com::read_guest(desc_va, g)) {
        note_bad_argument("CreateComputePipelineState: unreadable descriptor");
        return kEInvalidArg;
    }
    return create_compute_pso(g, riid, out);
}

HResult Device::create_compute_pso(const GuestComputePipelineStateDesc& g, const Guid& riid, OutPtr out) {
    com::Ref<RootSignature> rs = root_signature_ref(g.pRootSignature);
    if (g.pRootSignature != 0 && !rs) {
        note_bad_argument("CreateComputePipelineState: invalid root signature " + hex(g.pRootSignature));
        return kEInvalidArg;
    }
    pso::D3D12ComputePsoDesc d{};
    DxilDigest cs_digest{};
    NativeStageRecord embedded_cs;
    if (!read_shader(*this, g.CS, "CS", &d.cs, &cs_digest, &embedded_cs, xbox::EmbeddedStage::Cs)) {
        return kEInvalidArg;
    }
    if (d.cs == 0) {
        note_bad_argument("CreateComputePipelineState: no compute shader");
        return kEInvalidArg;
    }
    NativeStageRecord native_cs{};
    static const DxilDigest kZero{};
    if (native_index_size() != 0 && cs_digest != kZero) {
        count(find_native_stage(cs_digest, &native_cs) ? xbox::Counter::DxilStagesJoined
                                                       : xbox::Counter::DxilStagesNotJoined);
    }
    if (native_cs.identity == 0 && embedded_cs.identity != 0) {
        native_cs = embedded_cs;
        count(xbox::Counter::DxilStagesEmbeddedNative);
    }
    if (const ShaderPackIndex* pack = options_.shader_pack.get(); pack != nullptr) {
        uint64_t pack_key = 0;
        for (uint64_t k : {native_cs.identity != 0 ? native_cs.code_identity : uint64_t{0}, d.cs}) {
            auto it = k != 0 ? pack->find(k) : pack->end();
            if (it != pack->end() && it->second == PackStage::Compute) {
                pack_key = k;
                break;
            }
        }
        // Reflection join (G62) for the compute stage.
        bridge::PipelineLayoutDesc reflected_layout;
        bool reflected = false;
        if (const StageReflectionIndex* refl = options_.stage_reflection.get(); refl != nullptr && pack_key != 0) {
            if (auto rit = refl->find(pack_key); rit != refl->end()) {
                std::string report;
                JoinStats join_stats;
                const bool ok = static_cast<bool>(rs) &&
                                join_pipeline_layout(rs->signature().desc(), /*compute=*/true,
                                                     {{GpuShaderStageBits::Compute, &rit->second}}, &reflected_layout,
                                                     &report, &join_stats,
                                                     JoinOptions{options_.bridge.binding_arrays});
                count_work(Work::ReflectionLaneIndexedUnbound, join_stats.lane_indexed_bindings);
                if (ok) {
                    reflected = true;
                    count_work(Work::ReflectionTableSlotsOutsideRs, join_stats.table_slots_outside_rs);
                    count_work(Work::ReflectionRootDescriptorTables, join_stats.root_descriptor_tables);
                    count_work(Work::ReflectionUnresolvedBindings, join_stats.unresolved_bindings);
                } else {
                    const uint64_t rs_hash = rs ? rs->signature().hash() : 0;
                    std::lock_guard<std::mutex> lock(pack_mutex_);
                    if (reflection_failures_.insert({pack_key, 0, rs_hash}).second) {
                        char head[80];
                        std::snprintf(head, sizeof(head), "d3d12_xs: reflection join failed (compute 0x%016" PRIx64 "):\n",
                                      pack_key);
                        os::log(std::string(head) + (rs ? report : std::string("no root signature\n")));
                    }
                    pack_key = 0;  // not resolved: dispatches with it are skipped
                }
            }
        }
        if (pack_key != 0) {
            std::lock_guard<std::mutex> lock(pack_mutex_);
            const uint32_t module_id = pack_module_locked(pack_key, GpuShaderStage::Compute);
            pipeline_prewarm().note_title_pipeline(kPrewarmKindCompute, pack_key, 0);
            register_shader_module(d.cs, module_id);
            if (rs && pack_layouts_.insert({d.cs, 0}).second) {
                bridge::PipelineLayoutDesc layout =
                    reflected ? std::move(reflected_layout) : layout_from_root_signature(rs->signature().desc());
                for (bridge::GroupLayoutDesc& group : layout.groups) {
                    for (bridge::LayoutBinding& b : group.bindings) {
                        b.entry.visibility = GpuShaderStageBits::Compute;
                    }
                }
                bridge_->register_pipeline_layout(d.cs, 0, std::move(layout));
            }
        }
    }
    d.root_signature_id = rs ? static_cast<uint32_t>(rs->signature().hash()) : 0u;
    auto state = com::create_object<PipelineState>(com::implements(kID3D12PipelineState), device_ref(this), rs);
    if (!state) {
        return kEOutOfMemory;
    }
    state->initialize_guest_payload();
    state->compute_ = bridge_->create_compute_pipeline_state(d, rs ? &rs->signature() : nullptr);
    state->identities_ = {d.cs};
    state->resolved_ = (state->compute_->diagnostics() & bridge::BridgeDiag::UnknownShaderModule) == 0;
    if (!state->resolved_ && !shader_known(d.cs)) {
        note_unresolved_shader(d.cs, ShaderStage::Compute);
    }
    if (dump_) {
        state->dump_index_ = dump_->compute_pipeline(d.cs, cs_digest, native_cs.code_identity, native_cs.identity,
                                                     rs.get(), rs ? &rs->signature().desc() : nullptr);
    }
    return hand_out(std::move(state), riid, out);
}

// ---------------------------------------------------------------------------------------------
// Vtables
// ---------------------------------------------------------------------------------------------

namespace {

using ResM = Methods<IResourceHost>;
constexpr com::MethodDesc kResourceMethods[] = {
    ResM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),
    ResM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),
    ResM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),
    ResM::method<&IObjectHost::SetName>("SetName"),
    ResM::method<&IDeviceChildHost::GetDevice>("GetDevice"),
    ResM::method<&IResourceHost::Map>("Map"),                                   // 8
    ResM::method<&IResourceHost::Unmap>("Unmap"),                               // 9
    ResM::sret<&IResourceHost::GetDesc>("GetDesc"),                             // 10
    ResM::method<&IResourceHost::GetGPUVirtualAddress>("GetGPUVirtualAddress"), // 11
    ResM::method<&IResourceHost::WriteToSubresource>("WriteToSubresource"),     // 12
    ResM::method<&IResourceHost::ReadFromSubresource>("ReadFromSubresource"),   // 13
    ResM::method<&IResourceHost::GetHeapProperties>("GetHeapProperties"),       // 14
    ResM::stub("GetProtectedResourceSession"),                                  // 15 (ID3D12Resource1)
    ResM::sret<&IResourceHost::GetDesc1>("GetDesc1"),                           // 16 (ID3D12Resource2)
};
constexpr Guid kResourceIids[] = {
    IID_ID3D12Object, IID_ID3D12DeviceChild, IID_ID3D12Pageable,
    com::make_guid("9d5e227a-4430-4161-88b3-3eca6bb16e19"),  // ID3D12Resource1
    com::make_guid("be36ec3b-ea85-4aeb-a45a-e9d76404a495"),  // ID3D12Resource2
};

using HeapObjM = Methods<IHeapHost>;
constexpr com::MethodDesc kHeapMethods[] = {
    HeapObjM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),
    HeapObjM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),
    HeapObjM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),
    HeapObjM::method<&IObjectHost::SetName>("SetName"),
    HeapObjM::method<&IDeviceChildHost::GetDevice>("GetDevice"),
    HeapObjM::sret<&IHeapHost::GetDesc>("GetDesc"),  // 8
};
constexpr Guid kHeapIids[] = {IID_ID3D12Object, IID_ID3D12DeviceChild, IID_ID3D12Pageable,
                              com::make_guid("572f7389-2168-49e3-9693-d6df5871bf6d")};  // ID3D12Heap1

using PsoM = Methods<IPipelineStateHost>;
constexpr com::MethodDesc kPipelineStateMethods[] = {
    PsoM::method<&IObjectHost::GetPrivateData>("GetPrivateData"),
    PsoM::method<&IObjectHost::SetPrivateData>("SetPrivateData"),
    PsoM::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),
    PsoM::method<&IObjectHost::SetName>("SetName"),
    PsoM::method<&IDeviceChildHost::GetDevice>("GetDevice"),
    PsoM::method<&IPipelineStateHost::GetCachedBlob>("GetCachedBlob"),  // 8
};
constexpr Guid kPageable[] = {IID_ID3D12Object, IID_ID3D12DeviceChild, IID_ID3D12Pageable};

constexpr com::InterfaceOptions opts(uint32_t slots, std::span<const Guid> extra) {
    com::InterfaceOptions o;
    o.slot_count = slots;
    o.policy = com::SlotPolicy::ReturnNotImpl;
    o.extra_iids = extra;
    return o;
}

}  // namespace

const com::Interface<IResourceHost> kID3D12Resource = com::declare_interface<IResourceHost>(
    "ID3D12Resource", IID_ID3D12Resource, kResourceMethods, opts(24, kResourceIids));
const com::Interface<IHeapHost> kID3D12Heap =
    com::declare_interface<IHeapHost>("ID3D12Heap", IID_ID3D12Heap, kHeapMethods, opts(16, kHeapIids));
const com::Interface<IPipelineStateHost> kID3D12PipelineState = com::declare_interface<IPipelineStateHost>(
    "ID3D12PipelineState", IID_ID3D12PipelineState, kPipelineStateMethods, opts(16, kPageable));

bool va_watch_resource(const char* kind, const Resource* r, const std::string& detail, uint64_t handle) {
    if (r == nullptr || !va_watch::active()) {
        return false;
    }
    const uint64_t size = r->is_buffer() ? r->desc().Width : committed_memory_size(r->desc());
    const uint64_t va = r->is_buffer() ? r->gpu_va() : r->descriptor_va();
    if (!va_watch::touch(kind, va, size, "resource id " + std::to_string(r->id()) + ", " + detail)) {
        return false;
    }
    va_watch::watch_handle(handle, va);
    return true;
}

}  // namespace playgta6::gpu::d3d12_guest
