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

  # Every sha256 below is generated, not hand-written: on each release,
  # .github/workflows/release.yml's update-packaging job runs
  # scripts/update-packaging.sh, which reads the hashes out of the release's
  # own published SHA256SUMS asset. They are therefore the hashes of the
  # artifacts users actually download, not of a local rebuild that merely
  # ought to match. Do not edit them by hand — re-run the script.
  REL = "https://github.com/Heniokhos-Systems/MLRift/releases/download/v#{version}".freeze
  RAW = "https://raw.githubusercontent.com/Heniokhos-Systems/MLRift/v#{version}".freeze

  on_macos do
    on_arm do
      url "#{REL}/mlrc-macos-arm64"
      sha256 "6f86d83f73cfb0569f39e7172284c8ff43af738257eb1d9887d3785f215e0d8a"
      resource "mlr" do
        url "#{REL}/mlr-macos-arm64"
        sha256 "702b4f7a9aa2855324031986c1c11b1ff2a7814450022d4e5db528a2216bc0d1"
      end
    end
    on_intel do
      url "#{REL}/mlrc-macos-x86_64"
      sha256 "05e5b99c1baa89548fade50aef19b0fac1c9494d42d72e8040130a3e7957c870"
      resource "mlr" do
        url "#{REL}/mlr-macos-x86_64"
        sha256 "ff0af457a32694d23802f713be8dd9bae61e823bc93da0e1a36e10abcf42c6e5"
      end
    end
  end

  on_linux do
    on_arm do
      url "#{REL}/mlrc-linux-arm64"
      sha256 "b862f3f1d729905044405de4cc5cc53af03891f7f8ee7d4f3da1079eaea3379b"
      resource "mlr" do
        url "#{REL}/mlr-linux-arm64"
        sha256 "28fa5ed7c64178d8adde25780ab316c3514dd7b955659a94a0c7e1a547d0091e"
      end
    end
    on_intel do
      url "#{REL}/mlrc-linux-x86_64"
      sha256 "6a36aa4dc946b6b3ae23b363f5d8b6230b7742cdca18817aae2ee333683f0999"
      resource "mlr" do
        url "#{REL}/mlr-linux-x86_64"
        sha256 "7255594d41898e9cd4beaa01c26387e4aaedee284baf000fbd4f2fbaca75a3db"
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
    # mlrc searches /usr/local/share/mlrift/, /usr/share/mlrift/,
    # $HOME/.local/share/mlrift/ and both Homebrew prefixes
    # (/opt/homebrew/share/mlrift/ on Apple Silicon,
    # /home/linuxbrew/.linuxbrew/share/mlrift/ on Linux), so share/mlrift is
    # on the search path for every Homebrew layout. Those paths said
    # "kernrift" until v1.1.0 — a leftover from the fork that meant no
    # OS-level install could resolve `import "std/..."` at all.
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
    # formula installed std/ somewhere mlrc actually searches. It is also
    # load-bearing as a regression test — the search paths said "kernrift"
    # until v1.1.0, and a failed import used to emit a binary and exit 0
    # rather than failing, so this assertion would have passed while the
    # stdlib was entirely unreachable.
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
