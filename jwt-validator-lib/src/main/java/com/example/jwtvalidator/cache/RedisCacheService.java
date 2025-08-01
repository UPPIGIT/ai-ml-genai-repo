package com.example.jwtvalidator.cache;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import com.example.jwtvalidator.cache.serializer.Serializer;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

/**
 * A generic Redis-backed cache service implementation.
 * Stores and retrieves objects using a pluggable serializer and expiry policy.
 *
 * @param <T> the type of object to cache
 */
public class RedisCacheService<T> implements CacheService<T> {
    private static final Logger logger = LoggerFactory.getLogger(RedisCacheService.class);
    private final StringRedisTemplate redisTemplate;
    private final Serializer<T> serializer;
    private final ExpiryPolicy<T> expiryPolicy;
    private final String keyPrefix;

    /**
     * Construct a RedisCacheService with a static expiry policy.
     *
     * @param redisTemplate the Redis template
     * @param serializer the serializer for objects
     * @param cacheExpiry the expiry time in seconds for all keys
     * @param keyPrefix the prefix to use for all Redis keys
     */
    public RedisCacheService(StringRedisTemplate redisTemplate, Serializer<T> serializer, long cacheExpiry, String keyPrefix) {
        this(redisTemplate, serializer, (key, value) -> cacheExpiry, keyPrefix);
    }

    /**
     * Construct a RedisCacheService with a custom expiry policy.
     *
     * @param redisTemplate the Redis template
     * @param serializer the serializer for objects
     * @param expiryPolicy the expiry policy for keys/values
     * @param keyPrefix the prefix to use for all Redis keys
     */
    public RedisCacheService(StringRedisTemplate redisTemplate, Serializer<T> serializer, ExpiryPolicy<T> expiryPolicy, String keyPrefix) {
        this.redisTemplate = redisTemplate;
        this.serializer = serializer;
        this.expiryPolicy = expiryPolicy;
        this.keyPrefix = keyPrefix != null ? keyPrefix : "";
    }

    /**
     * Retrieve a value from Redis by key.
     *
     * @param key the cache key
     * @return an Optional containing the value if present, otherwise empty
     */
    @Override
    public Optional<T> get(String key) {
        try {
            String redisKey = keyPrefix + key;
            logger.debug("[RedisCacheService] Attempting to get value from Redis with key: {}", redisKey);
            String json = redisTemplate.opsForValue().get(redisKey);
            if (json != null) {
                logger.debug("[RedisCacheService] Cache hit for key: {}", redisKey);
                return Optional.of(serializer.deserialize(json));
            } else {
                logger.debug("[RedisCacheService] Cache miss for key: {}", redisKey);
            }
        } catch (Exception e) {
            logger.error("[RedisCacheService] Error getting value from Redis for key: {}: {}", key, e.getMessage(), e);
        }
        return Optional.empty();
    }

    /**
     * Store a value in Redis by key, using the configured expiry policy.
     *
     * @param key the cache key
     * @param value the value to cache
     */
    @Override
    public void put(String key, T value) {
        try {
            String redisKey = keyPrefix + key;
            logger.debug("[RedisCacheService] Putting value into Redis with key: {}", redisKey);
            String json = serializer.serialize(value);
            long expiry = expiryPolicy.getExpirySeconds(key, value);
            redisTemplate.opsForValue().set(redisKey, json, expiry, TimeUnit.SECONDS);
            logger.debug("[RedisCacheService] Successfully put value into Redis with key: {} (expiry: {} seconds)", redisKey, expiry);
        } catch (Exception e) {
            logger.error("[RedisCacheService] Error putting value into Redis for key: {}: {}", key, e.getMessage(), e);
        }
    }
}
