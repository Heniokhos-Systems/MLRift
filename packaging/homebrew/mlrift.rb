# typed: false
# frozen_string_literal: true

# Homebrew formula for MLRift. Lives in the tap Heniokhos-Systems/homebrew-mlrift
# as Formula/mlrift.rb, so users run:
#   brew install heniokhos-systems/mlrift/mlrift
class Mlrift < Formula
  desc "Self-hosted systems language and compiler for machine-learning workloads"
  homepage "https://github.com/Heniokhos-Systems/MLRift"
  version "1.1.0"
  license "Apache-2.0"

  # PLACEHOLDER — v1.1.0 has not been released yet. Every sha256 below is
  # rewritten by scripts/update-packaging.sh once build-release publishes
  # SHA256SUMS for this tag; do not push this formula to the tap as-is.
  REL = "https://github.com/Heniokhos-Systems/MLRift/releases/download/v#{version}".freeze
  RAW = "https://raw.githubusercontent.com/Heniokhos-Systems/MLRift/v#{version}".freeze

  on_macos do
    on_arm do
      url "#{REL}/mlrc-macos-arm64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      resource "mlr" do
        url "#{REL}/mlr-macos-arm64"
        sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      end
    end
    on_intel do
      url "#{REL}/mlrc-macos-x86_64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      resource "mlr" do
        url "#{REL}/mlr-macos-x86_64"
        sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      end
    end
  end

  on_linux do
    on_arm do
      url "#{REL}/mlrc-linux-arm64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      resource "mlr" do
        url "#{REL}/mlr-linux-arm64"
        sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      end
    end
    on_intel do
      url "#{REL}/mlrc-linux-x86_64"
      sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      resource "mlr" do
        url "#{REL}/mlr-linux-x86_64"
        sha256 "0000000000000000000000000000000000000000000000000000000000000000"
      end
    end
  end

  def install
    # The stable download IS the mlrc binary.
    bin.install stable.url.split("/").last => "mlrc"

    # mlr runner (checksum-verified via the resource above).
    resource("mlr").stage { bin.install Dir["*"].first => "mlr" }

    # Standard library, installed under Homebrew's prefix (share/mlrift).
    #
    # KNOWN BUG: as of v1.1.0, mlrc's Linux/macOS stdlib search paths are
    # still hardcoded in src/main.mlr to /usr/local/share/kernrift/,
    # /usr/share/kernrift/ and $HOME/.local/share/kernrift/ -- a leftover
    # from the KernRift fork that was never renamed to mlrift, and there is
    # no MLR_STDLIB override. share/mlrift is therefore NOT currently on
    # mlrc's search path; `import "std/..."` will not resolve until that is
    # fixed upstream. Installing here anyway to match install.sh's (also
    # currently broken) convention, so the fix is a one-line rename away
    # from making every packaging channel work at once.
    std = share/"mlrift/std"
    std.mkpath
    %w[
      alloc string io math math_float fmt mem memfast vec map
      color fb fixedpoint font widget time log net sha256
    ].each do |m|
      system "curl", "-fsSL", "-o", std/"#{m}.mlr", "#{RAW}/std/#{m}.mlr"
    end
  end

  test do
    # Importing a stdlib module is the real test: it only compiles if the
    # formula installed std/ where mlrc searches. NOTE: given the known bug
    # above, this test will currently FAIL against share/mlrift until the
    # search-path rename lands in src/main.mlr — kept as the target-state
    # test so it starts passing the moment that fix ships.
    (testpath/"t.mlr").write <<~MLR
      import "std/io.mlr"
      fn main() -> uint64 {
          return 0
      }
    MLR
    system bin/"mlrc", "t.mlr", "-o", "t.mlrbo"
    assert_predicate testpath/"t.mlrbo", :exist?
    assert_match "1.1.0", shell_output("#{bin}/mlrc --version")
  end
end
