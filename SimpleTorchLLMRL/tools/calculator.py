

import re
import math
from .tool import Tool


class Calculator(Tool):
    """Enhanced calculator tool that supports basic arithmetic operations."""

    def __init__(self) -> None:
        super().__init__(
            tags="calculator",
            description="""This is a calculator tool that performs basic arithmetic operations.
            Supported operations: +, -, *, /, **, %, sqrt(), sin(), cos(), tan(), log(), abs()
            Examples: 
            - <calculator>2+2</calculator>
            - <calculator>10*5-3</calculator>
            - <calculator>sqrt(16)</calculator>
            - <calculator>2**3</calculator>""",
            name="Calculator"
        )

    def _execute(self, input_str: str) -> str | None:
        """Execute a mathematical expression safely."""


        

        try:
            # Clean the input
            expression = input_str.strip()
            
            # Safety check - only allow safe mathematical operations
            if not self._is_safe_expression(expression):
                return "Error: Invalid or unsafe expression"
            
            # Replace mathematical functions with math module equivalents
            expression = self._prepare_expression(expression)
            
            # Evaluate the expression safely
            result = eval(expression, {"__builtins__": {}, "math": math, "abs": abs})
            
            # Format the result
            if isinstance(result, float):
                # Round to avoid floating point precision issues
                if result.is_integer():
                    return str(int(result))
                else:
                    return f"{result:.6g}"  # Use scientific notation for very large/small numbers
            else:
                return str(result)
                
        except ZeroDivisionError:
            return "Error: Division by zero"
        except ValueError as e:
            return f"Error: Invalid value - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _is_safe_expression(self, expression: str) -> bool:
        """Check if the expression contains only safe mathematical operations."""
        # Allow only numbers, basic operators, parentheses, and specific functions
        safe_pattern = r'^[0-9+\-*/().\s%**sqrtsincostandlogabs,]+$'
        
        # Check for dangerous keywords
        dangerous_keywords = [
            'import', 'exec', 'eval', 'open', 'file', '__', 'class', 'def',
            'lambda', 'input', 'raw_input', 'compile', 'reload', 'globals',
            'locals', 'vars', 'dir', 'hasattr', 'getattr', 'setattr', 'delattr'
        ]
        
        expression_lower = expression.lower()
        for keyword in dangerous_keywords:
            if keyword in expression_lower:
                return False
        
        return bool(re.match(safe_pattern, expression.replace(' ', '')))

    def _prepare_expression(self, expression: str) -> str:
        """Prepare the expression by replacing function names with math module calls."""
        # Replace mathematical functions
        replacements = {
            'sqrt(': 'math.sqrt(',
            'sin(': 'math.sin(',
            'cos(': 'math.cos(',
            'tan(': 'math.tan(',
            'log(': 'math.log(',
        }
        
        for old, new in replacements.items():
            expression = expression.replace(old, new)
        
        return expression
