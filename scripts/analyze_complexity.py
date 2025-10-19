#!/usr/bin/env python3
"""
Complexity and Importance Audit Script for Operation Lean
Analyzes LOC, Cyclomatic Complexity, and Fan-in metrics for lolly_v7/src
"""

import os
import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json


class ComplexityAnalyzer(ast.NodeVisitor):
    """Calculate cyclomatic complexity for Python code"""

    def __init__(self):
        self.complexity = defaultdict(int)
        self.current_function = None

    def visit_FunctionDef(self, node):
        """Visit function definitions"""
        func_name = f"{node.name}"
        self.current_function = func_name
        self.complexity[func_name] = 1  # Base complexity
        self.generic_visit(node)
        self.current_function = None

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions"""
        self.visit_FunctionDef(node)

    def visit_If(self, node):
        """Each if adds complexity"""
        if self.current_function:
            self.complexity[self.current_function] += 1
        self.generic_visit(node)

    def visit_While(self, node):
        """Each while adds complexity"""
        if self.current_function:
            self.complexity[self.current_function] += 1
        self.generic_visit(node)

    def visit_For(self, node):
        """Each for adds complexity"""
        if self.current_function:
            self.complexity[self.current_function] += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Each except adds complexity"""
        if self.current_function:
            self.complexity[self.current_function] += 1
        self.generic_visit(node)

    def visit_With(self, node):
        """Each with adds complexity"""
        if self.current_function:
            self.complexity[self.current_function] += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        """Each boolean operation adds complexity"""
        if self.current_function:
            self.complexity[self.current_function] += len(node.values) - 1
        self.generic_visit(node)


def count_lines(file_path: str) -> int:
    """Count lines of code (excluding blanks and comments)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        loc = 0
        in_docstring = False
        for line in lines:
            stripped = line.strip()

            # Skip blank lines
            if not stripped:
                continue

            # Handle docstrings
            if '"""' in stripped or "'''" in stripped:
                in_docstring = not in_docstring
                continue

            if in_docstring:
                continue

            # Skip comments
            if stripped.startswith('#'):
                continue

            loc += 1

        return loc
    except Exception as e:
        print(f"Error counting lines in {file_path}: {e}", file=sys.stderr)
        return 0


def analyze_file_complexity(file_path: str) -> Dict:
    """Analyze a single Python file for complexity metrics"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        tree = ast.parse(code)
        analyzer = ComplexityAnalyzer()
        analyzer.visit(tree)

        loc = count_lines(file_path)
        max_cc = max(analyzer.complexity.values()) if analyzer.complexity else 1
        avg_cc = sum(analyzer.complexity.values()) / len(analyzer.complexity) if analyzer.complexity else 1

        return {
            'file': file_path,
            'loc': loc,
            'max_cc': max_cc,
            'avg_cc': round(avg_cc, 2),
            'function_count': len(analyzer.complexity),
            'functions': dict(analyzer.complexity)
        }
    except SyntaxError as e:
        print(f"Syntax error in {file_path}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
        return None


def calculate_fan_in(src_dir: str) -> Dict[str, int]:
    """Calculate fan-in (import count) for each file"""
    fan_in = defaultdict(int)
    src_path = Path(src_dir)

    for py_file in src_path.rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        fan_in[alias.name] += 1
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        fan_in[node.module] += 1

        except Exception as e:
            print(f"Error analyzing imports in {py_file}: {e}", file=sys.stderr)

    return dict(fan_in)


def main():
    src_dir = Path(__file__).parent.parent / 'src'

    if not src_dir.exists():
        print(f"Error: {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("OPERATION LEAN: Complexity and Importance Audit")
    print("=" * 80)
    print()

    # Analyze all Python files
    all_files = []
    for py_file in src_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue

        result = analyze_file_complexity(str(py_file))
        if result:
            all_files.append(result)

    print(f"Analyzed {len(all_files)} Python files")
    print()

    # Sort by different metrics
    by_loc = sorted(all_files, key=lambda x: x['loc'], reverse=True)[:20]
    by_max_cc = sorted(all_files, key=lambda x: x['max_cc'], reverse=True)[:20]

    # Calculate fan-in
    print("Calculating fan-in metrics...")
    fan_in = calculate_fan_in(str(src_dir))

    # Map fan-in to files
    file_fan_in = []
    for file_data in all_files:
        rel_path = str(Path(file_data['file']).relative_to(src_dir.parent))
        module_path = rel_path.replace('/', '.').replace('.py', '')

        # Check various module path formats
        possible_paths = [
            module_path,
            module_path.replace('src.', ''),
            Path(file_data['file']).stem,
        ]

        total_fan_in = sum(fan_in.get(p, 0) for p in possible_paths)

        if total_fan_in > 0:
            file_fan_in.append({
                'file': file_data['file'],
                'fan_in': total_fan_in
            })

    by_fan_in = sorted(file_fan_in, key=lambda x: x['fan_in'], reverse=True)[:20]

    # Output results
    print()
    print("=" * 80)
    print("TOP 20 FILES BY LINES OF CODE (LOC)")
    print("=" * 80)
    for i, f in enumerate(by_loc, 1):
        rel_path = Path(f['file']).relative_to(src_dir.parent)
        print(f"{i:2}. {f['loc']:5} LOC | {rel_path}")

    print()
    print("=" * 80)
    print("TOP 20 FILES BY MAX CYCLOMATIC COMPLEXITY")
    print("=" * 80)
    for i, f in enumerate(by_max_cc, 1):
        rel_path = Path(f['file']).relative_to(src_dir.parent)
        print(f"{i:2}. CC={f['max_cc']:3} | {f['loc']:5} LOC | {rel_path}")

        # Show top complex functions
        top_funcs = sorted(f['functions'].items(), key=lambda x: x[1], reverse=True)[:3]
        for func_name, cc in top_funcs:
            print(f"       └─ {func_name}(): CC={cc}")

    print()
    print("=" * 80)
    print("TOP 20 FILES BY FAN-IN (IMPORT COUNT)")
    print("=" * 80)
    for i, f in enumerate(by_fan_in, 1):
        rel_path = Path(f['file']).relative_to(src_dir.parent)
        print(f"{i:2}. Fan-in={f['fan_in']:3} | {rel_path}")

    # Save raw data for importance matrix
    output = {
        'by_loc': by_loc,
        'by_max_cc': by_max_cc,
        'by_fan_in': by_fan_in,
        'all_files': all_files
    }

    output_path = Path(__file__).parent / 'complexity_data.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Raw data saved to: {output_path}")
    print()


if __name__ == '__main__':
    main()
