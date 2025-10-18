"""Simple test to verify ContextManager integration with AnthropicLLM"""

import sys
from unittest.mock import Mock, MagicMock

# Mock the anthropic module before importing AnthropicLLM
sys.modules['anthropic'] = MagicMock()

from dinnovos.llms.anthropic import AnthropicLLM

def test_context_manager_initialization():
    """Test that ContextManager is properly initialized"""
    print("Testing ContextManager initialization...")
    
    # Create LLM instance (no API key needed for this test)
    try:
        llm = AnthropicLLM(
            api_key="test-key",
            model="claude-sonnet-4-5-20250929",
            max_tokens=50000,
            context_strategy="smart"
        )
        
        # Check that context_manager exists
        assert hasattr(llm, 'context_manager'), "context_manager attribute not found"
        assert llm.context_manager is not None, "context_manager is None"
        
        # Check context_manager properties
        assert llm.context_manager.max_tokens == 50000, "max_tokens not set correctly"
        assert llm.context_manager.strategy == "smart", "strategy not set correctly"
        assert llm.context_manager.reserve_tokens == 4096, "reserve_tokens not set correctly"
        
        print("✅ ContextManager initialized correctly")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_context_stats():
    """Test context statistics methods"""
    print("\nTesting context statistics...")
    
    try:
        llm = AnthropicLLM(
            api_key="test-key",
            max_tokens=10000,
            context_strategy="smart"
        )
        
        # Create test messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help you?"},
        ]
        
        # Get stats
        stats = llm.get_context_stats(messages)
        
        # Verify stats structure
        assert 'current_tokens' in stats, "current_tokens not in stats"
        assert 'max_tokens' in stats, "max_tokens not in stats"
        assert 'available_tokens' in stats, "available_tokens not in stats"
        assert 'usage_percent' in stats, "usage_percent not in stats"
        assert 'messages_count' in stats, "messages_count not in stats"
        assert 'within_limit' in stats, "within_limit not in stats"
        
        # Verify values
        assert stats['messages_count'] == 3, f"Expected 3 messages, got {stats['messages_count']}"
        assert stats['max_tokens'] == 10000, f"Expected max_tokens 10000, got {stats['max_tokens']}"
        assert stats['within_limit'] == True, "Messages should be within limit"
        
        print(f"✅ Context statistics working correctly")
        print(f"   - Current tokens: {stats['current_tokens']}")
        print(f"   - Usage: {stats['usage_percent']}%")
        print(f"   - Messages: {stats['messages_count']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_context_management():
    """Test context management with many messages"""
    print("\nTesting context management...")
    
    try:
        llm = AnthropicLLM(
            api_key="test-key",
            max_tokens=10000,  # Small limit to force truncation
            context_strategy="smart"
        )
        
        # Create many messages to exceed limit
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        # Add 100 messages with longer content
        for i in range(100):
            messages.append({
                "role": "user",
                "content": f"This is a long message number {i} with lots of content to fill up the context window and force truncation. " * 10
            })
            messages.append({
                "role": "assistant",
                "content": f"This is a long response to message {i} with additional content to increase token count. " * 10
            })
        
        # Get stats before management
        stats_before = llm.get_context_stats(messages)
        print(f"   Before: {stats_before['current_tokens']} tokens, {stats_before['messages_count']} messages")
        
        # Manually trigger context management
        managed_messages = llm.context_manager.manage(messages, verbose=False)
        
        # Get stats after management
        stats_after = llm.get_context_stats(managed_messages)
        print(f"   After: {stats_after['current_tokens']} tokens, {stats_after['messages_count']} messages")
        
        # Verify truncation occurred
        assert len(managed_messages) < len(messages), "Messages should have been truncated"
        assert stats_after['within_limit'] == True, "Managed messages should be within limit"
        
        print(f"✅ Context management working correctly")
        print(f"   - Truncated from {len(messages)} to {len(managed_messages)} messages")
        print(f"   - Saved {stats_before['current_tokens'] - stats_after['current_tokens']} tokens")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reset_stats():
    """Test resetting context statistics"""
    print("\nTesting reset statistics...")
    
    try:
        llm = AnthropicLLM(
            api_key="test-key",
            max_tokens=10000,
            context_strategy="smart"
        )
        
        # Create messages and trigger truncation
        messages = [{"role": "user", "content": "x" * 100000}]  # Much longer to exceed limit
        managed = llm.context_manager.manage(messages)
        
        # Check that stats were recorded
        stats = llm.get_context_stats(messages)
        assert stats['truncated_count'] > 0, "Truncation should have occurred"
        
        # Reset stats
        llm.reset_context_stats()
        
        # Verify stats were reset
        stats_after = llm.get_context_stats(messages)
        assert stats_after['truncated_count'] == 0, "truncated_count should be 0 after reset"
        assert stats_after['total_tokens_saved'] == 0, "total_tokens_saved should be 0 after reset"
        
        print("✅ Reset statistics working correctly")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("CONTEXT MANAGER INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        test_context_manager_initialization,
        test_context_stats,
        test_context_management,
        test_reset_stats
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
    else:
        print(f"❌ {total - passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
