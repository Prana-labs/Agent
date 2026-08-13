import uuid
import json
from typing import Optional, Dict, Any

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from state import sessions


class RedisSessionManager:
    """
    Session manager that uses Redis for persistence.
    Falls back to in-memory sessions if Redis is unavailable.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.redis_client = None
        self.use_redis = False
        self.fallback_to_memory = True
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True
                )
                self.redis_client.ping()
                self.use_redis = True
                print("✅ Redis connection established")
            except (redis.ConnectionError, Exception) as e:
                print(f"⚠️ Redis unavailable, falling back to memory: {e}")
                self.redis_client = None
        else:
            print("⚠️ redis package not installed, using memory fallback")
    
    def _get_key(self, thread_id: str, key: str) -> str:
        return f"session:{thread_id}:{key}"
    
    def create_session(self, thread_id: str, data: Dict[str, Any]) -> bool:
        """Save session data to Redis or memory."""
        try:
            if self.use_redis:
                key = self._get_key(thread_id, "data")
                self.redis_client.set(key, json.dumps(data))
            else:
                sessions[thread_id] = data
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            if not self.use_redis and self.fallback_to_memory:
                sessions[thread_id] = data
            return False
    
    def get_session(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data from Redis or memory."""
        try:
            if self.use_redis:
                key = self._get_key(thread_id, "data")
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
                return None
            else:
                return sessions.get(thread_id)
        except Exception as e:
            print(f"Error retrieving session: {e}")
            if self.fallback_to_memory:
                return sessions.get(thread_id)
            return None
    
    def delete_session(self, thread_id: str) -> bool:
        """Delete session data from Redis or memory."""
        try:
            if self.use_redis:
                keys = self.redis_client.keys(self._get_key(thread_id, "*"))
                if keys:
                    self.redis_client.delete(*keys)
            else:
                if thread_id in sessions:
                    del sessions[thread_id]
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            if self.fallback_to_memory and thread_id in sessions:
                del sessions[thread_id]
            return False
    
    def session_exists(self, thread_id: str) -> bool:
        """Check if session exists."""
        try:
            if self.use_redis:
                key = self._get_key(thread_id, "data")
                return self.redis_client.exists(key)
            else:
                return thread_id in sessions
        except Exception:
            return thread_id in sessions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection stats."""
        if self.use_redis and self.redis_client:
            info = self.redis_client.info()
            return {
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "memory_used": info.get("used_memory_human"),
            }
        return {"fallback": "memory", "active_sessions": len(sessions)}


# Global instance for easy import
redis_session_manager = RedisSessionManager()

# Convenience functions for compatibility with existing code
def create_session(thread_id: str, data: Dict[str, Any]) -> bool:
    return redis_session_manager.create_session(thread_id, data)

def get_session(thread_id: str) -> Optional[Dict[str, Any]]:
    return redis_session_manager.get_session(thread_id)

def delete_session(thread_id: str) -> bool:
    return redis_session_manager.delete_session(thread_id)

def session_exists(thread_id: str) -> bool:
    return redis_session_manager.session_exists(thread_id)
