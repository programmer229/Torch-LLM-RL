

from SimpleTorchLLMRL.chat.message import Message, MessageType
from SimpleTorchLLMRL.tools.calculator import Calculator






def test_basic_addition():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>2+2</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["4"])
    assert result == expected


def test_basic_multiplication():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>2*3</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["6"])
    assert result == expected



def test_basic_sqrt():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>sqrt(16)</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["4"])
    assert result == expected


def test_basic_subtraction():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>10-3</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["7"])
    assert result == expected


def test_basic_division():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>15/3</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["5"])
    assert result == expected


def test_power_operation():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>2**3</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["8"])
    assert result == expected


def test_modulo_operation():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>10%3</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["1"])
    assert result == expected


def test_complex_expression():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>2*3+4</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["10"])
    assert result == expected


def test_parentheses():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>(2+3)*4</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["20"])
    assert result == expected


def test_floating_point_division():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>10/3</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["3.33333"])
    assert result == expected


def test_absolute_value():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>abs(-5)</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["5"])
    assert result == expected


def test_trigonometric_sin():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>sin(0)</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["0"])
    assert result == expected


def test_trigonometric_cos():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>cos(0)</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["1"])
    assert result == expected


def test_natural_log():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>log(2.71828)</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    # log(e) should be approximately 1
    assert "1" in result


def test_division_by_zero():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>10/0</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["Error: Division by zero"])
    assert result == expected


def test_invalid_expression():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>invalid_function(5)</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["Error: Invalid or unsafe expression"])
    assert result == expected


def test_unsafe_expression():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>import os</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["Error: Invalid or unsafe expression"])
    assert result == expected


def test_multiple_calculations():
    calculator = Calculator()
    message = Message(
        content=f"Calculate: <{calculator._tags}>2+2</{calculator._tags}> and <{calculator._tags}>3*4</{calculator._tags}>",
        type=MessageType.MODEL
    )
    result = calculator(message)
    expected = calculator._format_str_output(["4", "12"])
    assert result == expected


def test_sqrt_of_negative():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}>sqrt(-1)</{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    assert "Error" in result and "domain error" in result


def test_empty_expression():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}></{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    assert "Error" in result


def test_whitespace_handling():
    calculator = Calculator()
    message = Message(content=f"<{calculator._tags}> 2 + 3 </{calculator._tags}>", type=MessageType.MODEL)
    result = calculator(message)
    expected = calculator._format_str_output(["5"])
    assert result == expected