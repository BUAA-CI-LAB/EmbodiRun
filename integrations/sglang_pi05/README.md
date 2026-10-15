# EmbodiRun SGLang Pi05 integration

This optional package owns the LeRobot Pi05 checkpoint compatibility pipeline.
Install it only in the managed SGLang inference environment. The managed entry
point is `embodirun-sglang-pi05-serve`. EmbodiRun's own
`embodirun-sglang-serve` launcher stays a thin wrapper in the root package.

The Pi05 pipeline exposes the shared `embodiinfer.async.v1` RTC metadata
contract. It keeps a per-session model-space action chunk and applies the
requested hard-prefix or soft-VJP denoising correction. RTC is currently
supported for SGLang's
monolithic worker (the normal single-4090 deployment); a split prefix/action
worker rejects an RTC request instead of silently running it synchronously.
