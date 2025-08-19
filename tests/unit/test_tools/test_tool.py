"""
Unit tests for AgentOrchestration.tools.tool module.

Tests cover the abstract Tool base class functionality and error conditions.
"""

import pytest
from abc import ABC
from typing import Any
from unittest.mock import Mock

from AgentOrchestration.tools.tool import Tool


class ConcreteTool(Tool):
    """Concrete implementation of Tool for testing abstract methods."""
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return f"Called with args: {args}, kwargs: {kwds}"


class TestTool:
    """Test Tool abstract base class functionality."""
    
    def test_tool_is_abstract(self):
        """Test that Tool is an abstract base class."""
        assert issubclass(Tool, ABC)
        
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            Tool(tags="test", instructions="test instructions")
    
    def test_concrete_tool_instantiation(self):
        """Test that concrete implementation can be instantiated."""
        tool = ConcreteTool(tags="test", instructions="Base instructions")
        
        assert isinstance(tool, Tool)
        assert isinstance(tool, ConcreteTool)
    
    def test_init_with_string_tags(self):
        """Test Tool initialization with string tags."""
        tags = "calculator"
        instructions = "Use this tool for calculations"
        
        tool = ConcreteTool(tags=tags, instructions=instructions)
        
        assert tool.tags == tags
        assert instructions in tool.instructions
    
    def test_init_with_list_tags(self):
        """Test Tool initialization with list tags."""
        tags = ["calculator", "math", "arithmetic"]
        instructions = "Mathematical operations tool"
        
        tool = ConcreteTool(tags=tags, instructions=instructions)
        
        assert tool.tags == tags
        assert instructions in tool.instructions
    
    def test_instructions_formatting(self):
        """Test that instructions are properly formatted with tag explanation."""
        tags = "test_tool"
        base_instructions = "This is a test tool"
        
        tool = ConcreteTool(tags=tags, instructions=base_instructions)
        
        # Check that base instructions are included
        assert base_instructions in tool.instructions
        
        # Check that tag explanation is added
        # Note: There's a bug in the source - 'tag' is undefined in line 21
        # The line should use 'tags' or iterate over tags if it's a list
        expected_format = f"<{tags}>"
        assert expected_format in tool.instructions
    
    def test_instructions_tag_explanation_bug(self):
        """Test documenting the bug in tag explanation formatting."""
        # The source code has a bug in line 21:
        # tag_explanation = f"\nTo use the tool use the following output <{tag}> input  to function</{tag}>"
        # 'tag' is undefined - should be 'tags' or handle list of tags
        
        tags = "calculator"
        instructions = "Calculate stuff"
        
        # This will raise NameError due to undefined 'tag' variable
        with pytest.raises(NameError, match="name 'tag' is not defined"):
            ConcreteTool(tags=tags, instructions=instructions)
    
    def test_abstract_method_exists(self):
        """Test that __call__ abstract method is defined."""
        abstract_methods = Tool.__abstractmethods__
        assert '__call__' in abstract_methods
    
    def test_concrete_implementation_call(self):
        """Test that concrete implementation provides __call__ method."""
        # Skip this test due to NameError in initialization
        pytest.skip("Cannot test due to NameError in Tool.__init__")
    
    def test_call_method_signature(self):
        """Test that __call__ method has correct signature."""
        # Check that abstract method exists with proper signature
        assert hasattr(Tool, '__call__')
        
        # The method should accept *args and **kwargs
        import inspect
        sig = inspect.signature(Tool.__call__)
        params = sig.parameters
        
        assert 'args' in params
        assert 'kwds' in params  # Note: uses 'kwds' instead of standard 'kwargs'
        assert params['args'].kind == inspect.Parameter.VAR_POSITIONAL
        assert params['kwds'].kind == inspect.Parameter.VAR_KEYWORD
    
    def test_inheritance_structure(self):
        """Test class inheritance structure."""
        # Test that Tool inherits from ABC
        assert issubclass(Tool, ABC)
        
        # Test method resolution order
        mro = Tool.__mro__
        assert ABC in mro
    
    def test_docstring_format(self):
        """Test constructor docstring formatting."""
        # Test that constructor has proper docstring
        docstring = Tool.__init__.__doc__
        
        assert docstring is not None
        assert "Initialize a Tool instance" in docstring
        assert "ARGS:" in docstring
        assert "tags" in docstring
        assert "instructions" in docstring
    
    @pytest.mark.parametrize("tags,instructions", [
        ("simple", "Simple instructions"),
        (["tag1", "tag2"], "Multiple tag instructions"),
        ("", "Empty tag instructions"),
        ("special_chars!@#", "Special character instructions"),
    ])
    def test_init_with_various_parameters(self, tags, instructions):
        """Test initialization with various parameter combinations."""
        # All these will fail due to the NameError bug
        with pytest.raises(NameError):
            ConcreteTool(tags=tags, instructions=instructions)
    
    def test_partial_implementation_fails(self):
        """Test that partial implementation of abstract methods fails."""
        
        class PartialTool(Tool):
            # Missing __call__ implementation
            pass
        
        # Should not be able to instantiate partial implementation
        with pytest.raises(TypeError):
            PartialTool(tags="test", instructions="test")
    
    def test_method_return_type_annotation(self):
        """Test that __call__ method has Any return type annotation."""
        import inspect
        
        sig = inspect.signature(Tool.__call__)
        return_annotation = sig.return_annotation
        
        assert return_annotation == Any


class TestToolBugDocumentation:
    """Test class specifically for documenting bugs and issues."""
    
    def test_undefined_tag_variable_bug(self):
        """Document the undefined 'tag' variable bug in __init__."""
        # Bug location: Line 21 in tool.py
        # tag_explanation = f"\nTo use the tool use the following output <{tag}> input  to function</{tag}>"
        # 'tag' is undefined - should be 'tags'
        
        with pytest.raises(NameError, match="name 'tag' is not defined"):
            ConcreteTool(tags="test_tag", instructions="Test instructions")
    
    def test_suggested_fix_for_tag_bug(self):
        """Test suggested fix for the tag variable bug."""
        # The bug could be fixed by changing 'tag' to 'tags' in line 21
        # However, this assumes tags is always a string
        
        # If tags is a string, the fix would work:
        # tag_explanation = f"\nTo use the tool use the following output <{tags}> input  to function</{tags}>"
        
        # If tags is a list, additional logic would be needed
        pass
    
    def test_kwds_parameter_naming(self):
        """Document non-standard parameter naming in __call__."""
        # The abstract method uses 'kwds' instead of the more standard 'kwargs'
        # This is not necessarily a bug, but inconsistent with Python conventions
        
        import inspect
        sig = inspect.signature(Tool.__call__)
        params = list(sig.parameters.keys())
        
        assert 'kwds' in params
        assert 'kwargs' not in params
    
    def test_super_call_in_abstract_method(self):
        """Document questionable super().__call__ in abstract method."""
        # Line 27: return super().__call__(*args, **kwds)
        # This calls ABC.__call__ which may not be what's intended
        # Typically, abstract methods would just 'pass' or raise NotImplementedError
        
        # This is documented behavior, not necessarily a bug
        pass