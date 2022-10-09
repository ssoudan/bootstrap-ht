// Copyright (c) 2022. Sebastien Soudan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http:www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Bootstrap Hypothesis Testing
//!
//! Check the [`prelude`] module for the public API.
use thiserror::Error;

/// The prelude module re-exports the most commonly used types and traits.
/// This is the public API. Enjoy!
pub mod prelude;

#[cfg(any(feature = "unstable", test))]
/// unstable bootstrap API
pub mod bootstrap;

#[cfg(not(any(feature = "unstable", test)))]
pub(crate) mod bootstrap;

#[cfg(any(feature = "unstable", test))]
/// unstable utils API
pub mod utils;

#[cfg(not(any(feature = "unstable", test)))]
pub(crate) mod utils;

/// The error type for this crate.
#[derive(Debug, Clone, Error)]
pub enum Error {
    /// NotEnoughSamples
    #[error("Not enough samples")]
    NotEnoughSamples,
}
