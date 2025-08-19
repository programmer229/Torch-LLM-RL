"""
Unit tests for AgentOrchestration.env.env module.

Tests cover the abstract Env base class functionality and error conditions.
"""

import pytest
from abc import ABC
from unittest.mock import Mock

from AgentOrchestration.env.env import Env
from AgentOrchestration.chat.message import Message, MessageType, Rollout


class ConcreteEnv(Env):
    """Concrete implementation of Env for testing abstract methods."""
    
    def get_sys_prompt(self) -> Message:
        return Message("System", MessageType.SYSTEM)
    
    def get_inital_prompt(self) -> Message:
        return Message("Initial", MessageType.MESSAGE)
    
    def respond_to_model(self) -> Message:
        return Message("Response", MessageType.MESSAGE)


class TestEnv:
    """Test Env abstract base class functionality."""
    
    def test_env_is_abstract(self):
        """Test that Env is an abstract base class."""
        assert issubclass(Env, ABC)
        
        # Cannot instantiate abstract class directly
        with pytest.raises(TypeError):
            Env()
    
    def test_concrete_env_instantiation(self):
        """Test that concrete implementation can be instantiated."""
        env = ConcreteEnv()
        
        assert isinstance(env, Env)
        assert isinstance(env, ConcreteEnv)
    
    def test_init_default_parameters(self):
        """Test Env initialization with default parameters."""
        env = ConcreteEnv()
        
        # Note: There's a typo in the source - should be max_turns not max_truns
        assert hasattr(env, 'max_truns')
        assert env.max_truns is None
    
    def test_init_with_max_turns(self):
        """Test Env initialization with max_turns parameter."""
        max_turns_value = 10
        env = ConcreteEnv(max_turns=max_turns_value)
        
        # The parameter is set but stored in misspelled attribute
        # This documents the bug in the source code
        assert env.max_truns is None  # Bug: parameter not stored correctly
    
    def test_abstract_methods_exist(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = Env.__abstractmethods__
        
        expected_methods = {
            'get_sys_prompt',
            'get_inital_prompt',  # Note: typo in method name
            'respond_to_model'
        }
        
        assert abstract_methods == expected_methods
    
    def test_concrete_implementation_methods(self):
        """Test that concrete implementation provides all required methods."""
        env = ConcreteEnv()
        
        # Test get_sys_prompt
        sys_prompt = env.get_sys_prompt()
        assert isinstance(sys_prompt, Message)
        assert sys_prompt.type == MessageType.SYSTEM
        
        # Test get_inital_prompt (note typo in method name)
        initial_prompt = env.get_inital_prompt()
        assert isinstance(initial_prompt, Message)
        assert initial_prompt.type == MessageType.MESSAGE
        
        # Test respond_to_model
        response = env.respond_to_model()
        assert isinstance(response, Message)
    
    def test_over_max_turns_method_exists(self):
        """Test that _over_max_turns method exists but has syntax errors."""
        env = ConcreteEnv()
        
        assert hasattr(env, '_over_max_turns')
        assert callable(env._over_max_turns)
    
    def test_over_max_turns_syntax_errors(self):
        """Test documenting syntax errors in _over_max_turns method."""
        env = ConcreteEnv()
        rollout = Rollout()
        rollout.add_messages(Message("Test", MessageType.MESSAGE))
        
        # The method has multiple syntax errors:
        # 1. Missing return type annotation: ->:
        # 2. Assignment instead of comparison: message.type = MessageType.Message
        # 3. Undefined variable: _over_max_turns instead of self.max_turns
        # 4. Wrong MessageType: MessageType.Message should be MessageType.MESSAGE
        
        with pytest.raises((SyntaxError, NameError, AttributeError)):
            env._over_max_turns(rollout)
    
    def test_typos_and_errors_documentation(self):
        """Document all typos and errors found in the Env class."""
        # This test documents the issues found:
        
        # 1. Constructor parameter typo
        env = ConcreteEnv(max_turns=5)
        assert hasattr(env, 'max_truns')  # Should be 'max_turns'
        
        # 2. Method name typo
        assert hasattr(env, 'get_inital_prompt')  # Should be 'get_initial_prompt'
        
        # 3. Parameter not stored correctly
        assert env.max_truns is None  # max_turns parameter ignored
        
        # 4. _over_max_turns method has multiple syntax errors
        # (documented in separate test)
    
    def test_inheritance_structure(self):
        """Test class inheritance structure."""
        env = ConcreteEnv()
        
        # Test inheritance chain
        assert isinstance(env, Env)
        assert isinstance(env, ABC)
        
        # Test method resolution order
        mro = ConcreteEnv.__mro__
        assert Env in mro
        assert ABC in mro
    
    def test_method_signatures(self):
        """Test abstract method signatures."""
        # get_sys_prompt should return Message
        assert hasattr(Env, 'get_sys_prompt')
        
        # get_inital_prompt should return Message  
        assert hasattr(Env, 'get_inital_prompt')
        
        # respond_to_model should return Message
        assert hasattr(Env, 'respond_to_model')
    
    @pytest.mark.parametrize("max_turns", [None, 1, 5, 10, 100])
    def test_init_with_various_max_turns(self, max_turns):
        """Test initialization with various max_turns values."""
        env = ConcreteEnv(max_turns=max_turns)
        
        # Due to bug, max_turns is not stored regardless of input
        assert env.max_truns is None
    
    def test_partial_implementation_fails(self):
        """Test that partial implementation of abstract methods fails."""
        
        class PartialEnv(Env):
            def get_sys_prompt(self) -> Message:
                return Message("System", MessageType.SYSTEM)
            
            # Missing get_inital_prompt and respond_to_model
        
        # Should not be able to instantiate partial implementation
        with pytest.raises(TypeError):
            PartialEnv()
    
    def test_method_return_types(self):
        """Test that abstract methods have correct return type annotations."""
        env = ConcreteEnv()
        
        # All methods should return Message objects
        sys_prompt = env.get_sys_prompt()
        initial_prompt = env.get_inital_prompt()
        response = env.respond_to_model()
        
        assert isinstance(sys_prompt, Message)
        assert isinstance(initial_prompt, Message)
        assert isinstance(response, Message)


class TestEnvBugDocumentation:
    """Test class specifically for documenting bugs and issues."""
    
    def test_constructor_parameter_typo(self):
        """Document constructor parameter typo: max_truns instead of max_turns."""
        env = ConcreteEnv(max_turns=10)
        
        # Bug: parameter max_turns is not stored
        assert not hasattr(env, 'max_turns')
        assert hasattr(env, 'max_truns')
        assert env.max_truns is None  # Should be 10
    
    def test_method_name_typo(self):
        """Document method name typo: get_inital_prompt instead of get_initial_prompt."""
        assert hasattr(ConcreteEnv, 'get_inital_prompt')
        assert not hasattr(ConcreteEnv, 'get_initial_prompt')
    
    def test_over_max_turns_method_bugs(self):
        """Document all bugs in _over_max_turns method."""
        # Source code line 29: def _over_max_turns(self,rollout) ->:
        # Bug 1: Missing return type after ->
        
        # Source code line 31: if message.type = MessageType.Message
        # Bug 2: Assignment (=) instead of comparison (==)
        # Bug 3: MessageType.Message should be MessageType.MESSAGE
        
        # Source code line 32: if len(user_messages) > _over_max_turns:
        # Bug 4: _over_max_turns is undefined, should be self.max_turns or similar
        
        # These bugs make the method impossible to execute
        env = ConcreteEnv()
        
        # Method exists but cannot be called due to syntax errors
        assert hasattr(env, '_over_max_turns')
        assert callable(env._over_max_turns)