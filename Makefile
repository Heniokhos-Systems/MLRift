# MLRift compiler (built on KernRift)
# Usage:
#   make              - build the compiler (self-hosts from build/mlrc)
#   make test         - run test suite
#   make install      - install to ~/.local/bin/mlrc
#   make dist         - create distribution binaries for all platforms
#   make clean        - remove build artifacts
#   make bootstrap    - verify self-host fixed point

INSTALL_DIR ?= $(HOME)/.local/bin
DIST_DIR = dist

SRCS = src/lexer.mlr src/ast.mlr src/parser.mlr src/codegen.mlr \
       src/codegen_aarch64.mlr src/ir.mlr src/ir_aarch64.mlr src/ir_hip.mlr \
       src/format_macho.mlr src/format_pe.mlr src/format_hip.mlr src/format_amdgpu.mlr src/format_amdgpu_megakernel.mlr src/format_elf_dyn.mlr src/dyn_sym_registry.mlr \
       src/format_archive.mlr src/format_android.mlr src/bcj.mlr src/analysis.mlr src/inliner.mlr src/living.mlr \
       src/runtime.mlr src/formatter.mlr src/main.mlr

.PHONY: all build mlr-runner test install dist clean bootstrap

all: build mlr-runner

build: build/mlrc

# Build the .mlrbo runner. runner.mlr references filter_aarch64_bcj /
# filter_x86_64_bcj from bcj.mlr, so the two must be concatenated before
# compile — otherwise the runner builds with unresolved BCJ calls and
# silently corrupts every extracted slice (entry-point bytes get
# clobbered, slice bus-errors at startup on the device).
#
# `kr` is a shell wrapper (packaging/kr.sh) that catches exit-120 from
# kr-bin and re-execs the extracted ./kr-exec — needed on Termux/Android
# where raw execve from app data dirs is SELinux-denied. Other hosts hit
# the wrapper's `exit $status` line as a no-op since the runner exec's
# the slice directly.
mlr-runner: build/kr build/kr-bin

build/kr-runner.mlr: src/runner.mlr src/bcj.mlr
	@mkdir -p build
	cat src/runner.mlr src/bcj.mlr > build/kr-runner.mlr

build/kr-bin: build/kr-runner.mlr build/mlrc
	./build/mlrc --arch=x86_64 build/kr-runner.mlr -o build/kr-bin
	chmod +x build/kr-bin
	@echo "Built build/kr-bin (host-native runner binary)"

build/kr: packaging/kr.sh
	@mkdir -p build
	cp packaging/kr.sh build/kr
	chmod +x build/kr
	@echo "Built build/kr (shell wrapper for kr-bin)"

build/mlrc.mlr: $(SRCS)
	@mkdir -p build
	cat $(SRCS) > build/mlrc.mlr

# Self-compile. build/mlrc is committed as the bootstrap.
build/mlrc: build/mlrc.mlr
	@if [ ! -x build/mlrc ]; then \
		echo "build/mlrc not found — clone must include the committed bootstrap."; \
		exit 1; \
	fi
	./build/mlrc --arch=x86_64 build/mlrc.mlr -o build/mlrc.new
	mv build/mlrc.new build/mlrc
	chmod +x build/mlrc

# Run test suite
test: build/mlrc
	@echo "=== Running MLRift test suite ==="
	@echo '#!/bin/bash' > /tmp/mlrc-test && echo 'exec ./build/mlrc --arch=x86_64 "$$@"' >> /tmp/mlrc-test && chmod +x /tmp/mlrc-test
	@KRC=/tmp/mlrc-test bash tests/run_tests.sh || true

# Verify self-host fixed point (stage3 == stage4)
bootstrap: build/mlrc
	@echo "=== Bootstrap verification ==="
	@cp build/mlrc.mlr /tmp/mlrc_bs_src.mlr
	@./build/mlrc --arch=x86_64 /tmp/mlrc_bs_src.mlr -o /tmp/mlrc3_bs 2>/dev/null
	@chmod +x /tmp/mlrc3_bs
	@/tmp/mlrc3_bs --arch=x86_64 /tmp/mlrc_bs_src.mlr -o /tmp/mlrc4_bs 2>/dev/null
	@if diff /tmp/mlrc3_bs /tmp/mlrc4_bs >/dev/null 2>&1; then \
		echo "PASS: fixed point at $$(wc -c < /tmp/mlrc3_bs) bytes"; \
	else \
		echo "FAIL: stage3 != stage4"; exit 1; \
	fi
	@rm -f /tmp/mlrc_bs_src.mlr /tmp/mlrc3_bs /tmp/mlrc4_bs

# Install as "mlrc" in INSTALL_DIR
install: build/mlrc
	@mkdir -p $(INSTALL_DIR)
	cp build/mlrc $(INSTALL_DIR)/mlrc
	chmod +x $(INSTALL_DIR)/mlrc
	@echo "Installed: $(INSTALL_DIR)/mlrc"
	@echo "Ensure $(INSTALL_DIR) is in your PATH"

# Distribution binaries
dist: build/mlrc
	@mkdir -p $(DIST_DIR)
	@echo "=== Building distribution ==="
	cp build/mlrc $(DIST_DIR)/mlrc-linux-x86_64
	chmod +x $(DIST_DIR)/mlrc-linux-x86_64
	@echo "  mlrc-linux-x86_64"
	./build/mlrc --arch=arm64 build/mlrc.mlr -o $(DIST_DIR)/mlrc-linux-arm64 2>/dev/null
	chmod +x $(DIST_DIR)/mlrc-linux-arm64
	@echo "  mlrc-linux-arm64"
	./build/mlrc --arch=x86_64 --emit=pe build/mlrc.mlr -o $(DIST_DIR)/mlrc-windows-x86_64.exe 2>/dev/null
	@echo "  mlrc-windows-x86_64.exe"
	./build/mlrc --arch=arm64 --emit=pe build/mlrc.mlr -o $(DIST_DIR)/mlrc-windows-arm64.exe 2>/dev/null
	@echo "  mlrc-windows-arm64.exe"
	./build/mlrc build/mlrc.mlr -o $(DIST_DIR)/mlrc.mlrbo 2>/dev/null
	@echo "  mlrc.mlrbo (fat binary, 8 slices)"
	cp build/mlrc.mlr $(DIST_DIR)/mlrc-source.mlr
	@echo "  mlrc-source.mlr"
	@echo ""
	@ls -la $(DIST_DIR)/
	@echo "=== Distribution complete ==="

clean:
	rm -f build/mlrc.new build/mlrc.mlr
	rm -rf $(DIST_DIR)
	rm -f a.out output.elf test_input.mlr
	rm -f *.elf *.out
	@echo "Cleaned (build/mlrc preserved — committed bootstrap)."
