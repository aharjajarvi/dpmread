//! Fused multiply-add the way the reference OpenCV build rounds.
//!
//! Python's OpenCV (CachyOS package) is compiled with `-march=x86-64-v3
//! -O3`: GCC contracts `a * b + c` into FMA in plain C++ loops, and the
//! dispatched AVX2 kernels use FMA intrinsics. Ops that must round like it
//! are written once with `madd::<FMA>` and wrapped by `fma_dispatch!`, which
//! picks the fused version when the CPU has FMA (x86-64 runtime check;
//! always on aarch64) so `mul_add` compiles to one instruction.

/// `a * b + c`, fused when `FMA`.
#[inline(always)]
pub(crate) fn madd<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `fma_dispatch!(pub fn name(args) -> R => impl_fn)` defines `name`, which
/// calls `impl_fn::<true>` (FMA enabled) or `impl_fn::<false>`. With
/// `fma_dispatch!(avx2 ...)` the fused version is compiled for AVX2 as well
/// when the CPU has it (wider vectors, same rounding); that helps the
/// filters but slows the per-point remap, so it is opt-in.
macro_rules! fma_dispatch {
    (avx2 $(#[$m:meta])* $vis:vis fn $name:ident($($a:ident: $t:ty),* $(,)?) -> $r:ty => $imp:ident) => {
        fma_dispatch!(@impl "avx2,fma" $(#[$m])* $vis fn $name($($a: $t),*) -> $r => $imp);
    };
    ($(#[$m:meta])* $vis:vis fn $name:ident($($a:ident: $t:ty),* $(,)?) -> $r:ty => $imp:ident) => {
        fma_dispatch!(@impl "fma" $(#[$m])* $vis fn $name($($a: $t),*) -> $r => $imp);
    };
    (@impl $feat:literal $(#[$m:meta])* $vis:vis fn $name:ident($($a:ident: $t:ty),* $(,)?) -> $r:ty => $imp:ident) => {
        $(#[$m])*
        $vis fn $name($($a: $t),*) -> $r {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "fma")]
                unsafe fn fused($($a: $t),*) -> $r {
                    $imp::<true>($($a),*)
                }
                #[target_feature(enable = $feat)]
                unsafe fn fused_wide($($a: $t),*) -> $r {
                    $imp::<true>($($a),*)
                }
                if std::is_x86_feature_detected!("fma") {
                    if $feat != "fma" && std::is_x86_feature_detected!("avx2") {
                        // SAFETY: the CPU supports AVX2 and FMA (checked at runtime).
                        return unsafe { fused_wide($($a),*) };
                    }
                    // SAFETY: the CPU supports FMA (checked at runtime).
                    return unsafe { fused($($a),*) };
                }
                $imp::<false>($($a),*)
            }
            #[cfg(target_arch = "aarch64")]
            {
                $imp::<true>($($a),*)
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                $imp::<false>($($a),*)
            }
        }
    };
}

/// `avx2_dispatch!(pub fn name(args) -> R => impl_fn)`: integer kernels
/// (no float rounding involved) compiled once more for AVX2 and picked at
/// run time; the results are identical.
macro_rules! avx2_dispatch {
    ($(#[$m:meta])* $vis:vis fn $name:ident$(<$(const $g:ident: $gt:ty),*>)?($($a:ident: $t:ty),* $(,)?) -> $r:ty => $imp:ident) => {
        $(#[$m])*
        $vis fn $name$(<$(const $g: $gt),*>)?($($a: $t),*) -> $r {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn wide$(<$(const $g: $gt),*>)?($($a: $t),*) -> $r {
                    $imp$(::<$($g),*>)?($($a),*)
                }
                if std::is_x86_feature_detected!("avx2") {
                    // SAFETY: the CPU supports AVX2 (checked at runtime).
                    return unsafe { wide$(::<$($g),*>)?($($a),*) };
                }
            }
            $imp$(::<$($g),*>)?($($a),*)
        }
    };
}

pub(crate) use avx2_dispatch;
pub(crate) use fma_dispatch;
