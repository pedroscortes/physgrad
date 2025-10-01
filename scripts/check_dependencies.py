#!/usr/bin/env python3
"""
PhysGrad Dependency Checker

Validates include dependencies, checks for circular dependencies,
and ensures proper header guards.
"""

import os
import re
import sys
from typing import Set, List, Dict, Tuple
from pathlib import Path

class DependencyChecker:
    def __init__(self, src_dir: str = "src"):
        self.src_dir = Path(src_dir)
        self.includes: Dict[str, Set[str]] = {}
        self.header_guards: Dict[str, str] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def run_checks(self) -> bool:
        """Run all dependency checks. Returns True if all checks pass."""
        print("=== PhysGrad Dependency Analysis ===\n")

        self.scan_files()
        self.check_header_guards()
        self.check_circular_dependencies()
        self.check_missing_includes()
        self.check_include_order()
        self.generate_report()

        return len(self.errors) == 0

    def scan_files(self):
        """Scan all source files and extract includes."""
        print("Scanning source files...")

        for ext in [".h", ".hpp", ".cpp", ".cu", ".cuh"]:
            for file_path in self.src_dir.rglob(f"*{ext}"):
                self.analyze_file(file_path)

        print(f"Analyzed {len(self.includes)} files\n")

    def analyze_file(self, file_path: Path):
        """Analyze a single file for includes and header guards."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            rel_path = str(file_path.relative_to(Path(".")))
            self.includes[rel_path] = set()

            # Extract includes
            include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
            for match in re.finditer(include_pattern, content):
                include_file = match.group(1)
                self.includes[rel_path].add(include_file)

            # Check header guards for header files
            if file_path.suffix in [".h", ".hpp", ".cuh"]:
                self.check_header_guard(file_path, content)

        except Exception as e:
            self.errors.append(f"Failed to analyze {file_path}: {e}")

    def check_header_guard(self, file_path: Path, content: str):
        """Check for proper header guards or #pragma once."""
        rel_path = str(file_path.relative_to(Path(".")))

        # Check for #pragma once
        if "#pragma once" in content:
            self.header_guards[rel_path] = "#pragma once"
            return

        # Check for traditional header guards
        ifndef_pattern = r'#ifndef\s+([A-Z_][A-Z0-9_]*)'
        define_pattern = r'#define\s+([A-Z_][A-Z0-9_]*)'
        endif_pattern = r'#endif'

        ifndef_matches = re.findall(ifndef_pattern, content)
        define_matches = re.findall(define_pattern, content)
        endif_matches = re.findall(endif_pattern, content)

        if ifndef_matches and define_matches and endif_matches:
            guard_name = ifndef_matches[0]
            if guard_name == define_matches[0]:
                self.header_guards[rel_path] = guard_name

                # Check if guard name follows convention
                expected_guard = self.generate_expected_guard(file_path)
                if guard_name != expected_guard:
                    self.warnings.append(
                        f"{rel_path}: Header guard '{guard_name}' doesn't follow convention. "
                        f"Expected: '{expected_guard}'"
                    )
            else:
                self.errors.append(f"{rel_path}: Mismatched header guard defines")
        else:
            self.errors.append(f"{rel_path}: Missing or incomplete header guards")

    def generate_expected_guard(self, file_path: Path) -> str:
        """Generate expected header guard name."""
        # Convert path to uppercase with underscores
        rel_path = str(file_path.relative_to(Path(".")))
        guard = rel_path.upper().replace("/", "_").replace(".", "_")
        return f"PHYSGRAD_{guard}_H"

    def check_header_guards(self):
        """Validate header guard consistency."""
        print("Checking header guards...")

        guard_names = {}
        for file_path, guard in self.header_guards.items():
            if guard != "#pragma once":
                if guard in guard_names:
                    self.errors.append(
                        f"Duplicate header guard '{guard}' in {file_path} and {guard_names[guard]}"
                    )
                else:
                    guard_names[guard] = file_path

        print(f"Checked {len(self.header_guards)} header files\n")

    def check_circular_dependencies(self):
        """Check for circular include dependencies."""
        print("Checking for circular dependencies...")

        def has_circular_dependency(file_path: str, visited: Set[str], path: List[str]) -> bool:
            if file_path in visited:
                cycle_start = path.index(file_path)
                cycle = " -> ".join(path[cycle_start:] + [file_path])
                self.errors.append(f"Circular dependency detected: {cycle}")
                return True

            visited.add(file_path)
            path.append(file_path)

            if file_path in self.includes:
                for include in self.includes[file_path]:
                    # Resolve include to actual file path
                    resolved_include = self.resolve_include(include, file_path)
                    if resolved_include and has_circular_dependency(resolved_include, visited.copy(), path.copy()):
                        return True

            return False

        for file_path in self.includes:
            has_circular_dependency(file_path, set(), [])

        print("Circular dependency check completed\n")

    def resolve_include(self, include: str, from_file: str) -> str:
        """Resolve include path to actual file."""
        # Handle relative includes
        if include.startswith("src/") or include.startswith("./"):
            candidate = include
        else:
            # Try relative to current file
            from_dir = os.path.dirname(from_file)
            candidate = os.path.join(from_dir, include)

        # Normalize path
        candidate = os.path.normpath(candidate)

        # Check if file exists in our analysis
        if candidate in self.includes:
            return candidate

        # Try with different extensions
        for ext in [".h", ".hpp", ".cuh"]:
            if os.path.splitext(candidate)[1] == "":
                candidate_with_ext = candidate + ext
                if candidate_with_ext in self.includes:
                    return candidate_with_ext

        return None

    def check_missing_includes(self):
        """Check for missing forward declarations and includes."""
        print("Checking for missing includes...")

        # This is a simplified check - in practice, you'd need a full C++ parser
        # for comprehensive analysis
        for file_path, includes in self.includes.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check for common patterns that might need includes
                self.check_stl_usage(file_path, content, includes)
                self.check_physics_types(file_path, content, includes)

            except Exception as e:
                self.warnings.append(f"Could not check includes for {file_path}: {e}")

        print("Missing includes check completed\n")

    def check_stl_usage(self, file_path: str, content: str, includes: Set[str]):
        """Check for STL usage without proper includes."""
        stl_patterns = {
            r'\bstd::vector\b': 'vector',
            r'\bstd::string\b': 'string',
            r'\bstd::shared_ptr\b': 'memory',
            r'\bstd::unique_ptr\b': 'memory',
            r'\bstd::array\b': 'array',
            r'\bstd::map\b': 'map',
            r'\bstd::unordered_map\b': 'unordered_map',
            r'\bstd::chrono\b': 'chrono',
            r'\bstd::thread\b': 'thread',
            r'\bstd::mutex\b': 'mutex',
            r'\bstd::condition_variable\b': 'condition_variable',
        }

        for pattern, header in stl_patterns.items():
            if re.search(pattern, content):
                if not any(header in inc for inc in includes):
                    self.warnings.append(
                        f"{file_path}: Uses {pattern} but doesn't include <{header}>"
                    )

    def check_physics_types(self, file_path: str, content: str, includes: Set[str]):
        """Check for physics-specific type usage."""
        physics_patterns = {
            r'\bconcepts::\w+': 'src/concepts/physics_concepts.h',
            r'\btype_traits::\w+': 'src/concepts/type_traits.h',
            r'\bPhysicsEngine\b': 'src/physics_engine.h',
            r'\bfloat[234]\b': 'src/common_types.h',
            r'\bCUDA_CALLABLE\b': 'cuda_runtime.h',
        }

        for pattern, header in physics_patterns.items():
            if re.search(pattern, content):
                if not any(header in inc for inc in includes):
                    self.warnings.append(
                        f"{file_path}: Uses {pattern} but doesn't include {header}"
                    )

    def check_include_order(self):
        """Check include order follows conventions."""
        print("Checking include order...")

        for file_path in self.includes:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                include_lines = []
                for i, line in enumerate(lines):
                    if line.strip().startswith('#include'):
                        include_lines.append((i, line.strip()))

                self.validate_include_order(file_path, include_lines)

            except Exception as e:
                self.warnings.append(f"Could not check include order for {file_path}: {e}")

        print("Include order check completed\n")

    def validate_include_order(self, file_path: str, include_lines: List[Tuple[int, str]]):
        """Validate that includes follow the correct order."""
        # Expected order:
        # 1. System headers (<>)
        # 2. Local headers ("")
        # 3. Within each group, alphabetical order

        system_includes = []
        local_includes = []

        for line_num, include_line in include_lines:
            if '<' in include_line:
                system_includes.append((line_num, include_line))
            elif '"' in include_line:
                local_includes.append((line_num, include_line))

        # Check if system includes come before local includes
        if system_includes and local_includes:
            last_system_line = system_includes[-1][0]
            first_local_line = local_includes[0][0]

            if last_system_line > first_local_line:
                self.warnings.append(
                    f"{file_path}: System includes should come before local includes"
                )

        # Check alphabetical order within groups
        self.check_alphabetical_order(file_path, "system", system_includes)
        self.check_alphabetical_order(file_path, "local", local_includes)

    def check_alphabetical_order(self, file_path: str, group_name: str,
                                includes: List[Tuple[int, str]]):
        """Check if includes in a group are in alphabetical order."""
        if len(includes) <= 1:
            return

        include_names = [inc[1] for inc in includes]
        sorted_names = sorted(include_names)

        if include_names != sorted_names:
            self.warnings.append(
                f"{file_path}: {group_name.capitalize()} includes are not in alphabetical order"
            )

    def generate_report(self):
        """Generate final report."""
        print("=== DEPENDENCY ANALYSIS RESULTS ===\n")

        print(f"Files analyzed: {len(self.includes)}")
        print(f"Header files with guards: {len(self.header_guards)}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}\n")

        if self.errors:
            print("ERRORS:")
            for error in self.errors:
                print(f"  ❌ {error}")
            print()

        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
            print()

        # Generate dependency graph
        self.generate_dependency_graph()

        if not self.errors:
            print("✅ All dependency checks passed!")
        else:
            print("❌ Dependency checks failed!")

    def generate_dependency_graph(self):
        """Generate a simple dependency graph."""
        print("=== DEPENDENCY GRAPH ===")

        # Calculate dependencies per file
        dependency_counts = {}
        reverse_dependencies = {}

        for file_path, includes in self.includes.items():
            dependency_counts[file_path] = len(includes)

            for include in includes:
                resolved = self.resolve_include(include, file_path)
                if resolved:
                    if resolved not in reverse_dependencies:
                        reverse_dependencies[resolved] = set()
                    reverse_dependencies[resolved].add(file_path)

        # Find files with most dependencies
        most_dependencies = sorted(dependency_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nFiles with most dependencies:")
        for file_path, count in most_dependencies:
            print(f"  {file_path}: {count} dependencies")

        # Find files with most reverse dependencies (most included)
        most_included = sorted(reverse_dependencies.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        print("\nMost included files:")
        for file_path, dependents in most_included:
            print(f"  {file_path}: included by {len(dependents)} files")

        print()

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        src_dir = sys.argv[1]
    else:
        src_dir = "src"

    if not os.path.exists(src_dir):
        print(f"Error: Source directory '{src_dir}' not found")
        sys.exit(1)

    checker = DependencyChecker(src_dir)
    success = checker.run_checks()

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()