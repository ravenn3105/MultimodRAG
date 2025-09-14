# rate_limiter.py
# Rate limiting utility for Cohere API calls

import time
import threading
from typing import Optional

class RateLimiter:
    """
    Token bucket rate limiter for API calls
    """
    def __init__(self, max_calls: int = 90, period: float = 60.0):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum API calls allowed per period
            period: Time period in seconds (default: 60 seconds)
        """
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls = []
        print(f"🚦 Rate limiter initialized: {max_calls} calls per {period} seconds")
    
    def acquire(self):
        """
        Acquire permission to make an API call
        Will block if rate limit would be exceeded
        """
        with self.lock:
            current_time = time.time()
            
            # Remove calls older than the period
            self.calls = [call_time for call_time in self.calls 
                         if call_time > current_time - self.period]
            
            # Check if we're at the limit
            if len(self.calls) >= self.max_calls:
                # Calculate how long to wait
                oldest_call = min(self.calls)
                sleep_time = self.period - (current_time - oldest_call) + 0.1  # Small buffer
                
                if sleep_time > 0:
                    print(f"⏳ Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                    
                    # Clean up old calls after sleeping
                    current_time = time.time()
                    self.calls = [call_time for call_time in self.calls 
                                 if call_time > current_time - self.period]
            
            # Record this call
            self.calls.append(current_time)
    
    def get_stats(self) -> dict:
        """Get current rate limiter statistics"""
        with self.lock:
            current_time = time.time()
            recent_calls = [call_time for call_time in self.calls 
                           if call_time > current_time - self.period]
            
            return {
                'calls_in_period': len(recent_calls),
                'max_calls': self.max_calls,
                'period': self.period,
                'remaining_calls': max(0, self.max_calls - len(recent_calls))
            }

# Global rate limiter instance for Cohere API
# Set to 90 calls per minute to leave some buffer for the 100 call limit
cohere_rate_limiter = RateLimiter(max_calls=90, period=60.0)