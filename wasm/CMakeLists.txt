cmake_minimum_required(VERSION 3.16)
project(PhysGradWasm)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if building with Emscripten
if(EMSCRIPTEN)
    message(STATUS "Building for WebAssembly with Emscripten")

    # Emscripten-specific settings
    set(CMAKE_EXECUTABLE_SUFFIX ".js")

    # Include directories
    include_directories(../src)
    include_directories(../include)

    # Source files
    set(WASM_SOURCES
        wasm_main.cpp
        ../src/wasm_bridge.h
    )

    # Create WASM executable
    add_executable(physgrad_wasm ${WASM_SOURCES})

    # Emscripten compile flags
    target_compile_options(physgrad_wasm PRIVATE
        -O3
        -ffast-math
        -DWASM_BUILD
        -msimd128                    # Enable SIMD
        -mrelaxed-simd              # Allow relaxed SIMD
        -matomics                   # Enable atomics for threading
        -mbulk-memory              # Enable bulk memory operations
        -msign-ext                 # Enable sign extension
    )

    # Emscripten link flags
    target_link_options(physgrad_wasm PRIVATE
        "SHELL:-s WASM=1"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        "SHELL:-s INITIAL_MEMORY=67108864"        # 64MB initial
        "SHELL:-s MAXIMUM_MEMORY=2147483648"      # 2GB maximum
        "SHELL:-s STACK_SIZE=8388608"             # 8MB stack
        "SHELL:-s TOTAL_STACK=16777216"           # 16MB total stack
        "SHELL:-s EXPORTED_FUNCTIONS=['_main']"
        "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap']"
        "SHELL:-s EXPORT_ALL=0"
        "SHELL:-s MODULARIZE=1"
        "SHELL:-s EXPORT_NAME='PhysGradModule'"
        "SHELL:-s USE_ES6_IMPORT_META=0"
        "SHELL:-s ENVIRONMENT='web,worker'"
        "SHELL:--bind"                            # Enable embind
        "SHELL:-s USE_PTHREADS=0"                # Disable threads for now
        "SHELL:-s ASSERTIONS=0"                   # Disable assertions in release
        "SHELL:-s SAFE_HEAP=0"                   # Disable safe heap checking
        "SHELL:-s DISABLE_EXCEPTION_CATCHING=0"  # Keep exception handling
        "SHELL:-s FILESYSTEM=0"                  # Disable file system
        "SHELL:-s FETCH=0"                       # Disable fetch API
        "SHELL:-s LEGACY_GL_EMULATION=0"         # Disable legacy GL
    )

    # Debug build options
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(physgrad_wasm PRIVATE
            -g4                      # Full debug info
            -DDEBUG_WASM
        )
        target_link_options(physgrad_wasm PRIVATE
            "SHELL:-s ASSERTIONS=1"
            "SHELL:-s SAFE_HEAP=1"
            "SHELL:-s STACK_OVERFLOW_CHECK=1"
            "SHELL:-g4"
        )
    endif()

    # Generate TypeScript definitions
    add_custom_command(TARGET physgrad_wasm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/physgrad.d.ts
        ${CMAKE_CURRENT_BINARY_DIR}/physgrad.d.ts
        COMMENT "Copying TypeScript definitions"
    )

    # Create HTML test page for development
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_link_options(physgrad_wasm PRIVATE
            "SHELL:--shell-file ${CMAKE_CURRENT_SOURCE_DIR}/shell.html"
        )
    endif()

else()
    message(STATUS "Building native version for testing")

    # Native build for testing
    include_directories(../src)
    include_directories(../include)

    set(NATIVE_SOURCES
        wasm_main.cpp
        ../src/wasm_bridge.h
    )

    add_executable(physgrad_native ${NATIVE_SOURCES})

    target_compile_options(physgrad_native PRIVATE
        -O3
        -DNATIVE_BUILD
        -march=native
        -ffast-math
    )

    # Find and link threading library
    find_package(Threads REQUIRED)
    target_link_libraries(physgrad_native Threads::Threads)

endif()

# Create package for distribution
if(EMSCRIPTEN)
    # Copy additional files for web distribution
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/package.json.in
        ${CMAKE_CURRENT_BINARY_DIR}/package.json
        @ONLY
    )

    # Install targets
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/physgrad_wasm.js
        ${CMAKE_CURRENT_BINARY_DIR}/physgrad_wasm.wasm
        ${CMAKE_CURRENT_BINARY_DIR}/physgrad.d.ts
        ${CMAKE_CURRENT_BINARY_DIR}/package.json
        DESTINATION ${CMAKE_INSTALL_PREFIX}
    )
endif()