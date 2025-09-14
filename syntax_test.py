#!/usr/bin/env python3

# Simple syntax check for the lesson generator
import ast

try:
    with open('app/core/lesson_generator.py', 'r', encoding='utf-8') as f:
        code = f.read()

    ast.parse(code)
    print("✅ Syntax is valid!")

except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
    print(f"Line {e.lineno}: {e.text}")
    print(f"Position: {' ' * (e.offset - 1)}^")

except Exception as e:
    print(f"❌ Error: {e}")