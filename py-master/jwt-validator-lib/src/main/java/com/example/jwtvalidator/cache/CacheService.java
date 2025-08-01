package com.example.jwtvalidator.cache;

import java.util.Optional;

/**
 * A generic cache service interface for storing and retrieving objects by key.
 * Implementations may use in-memory, distributed, or no-op strategies.
 *
 * @param <T> the type of object to cache
 */
public interface CacheService<T> {
    /**
     * Retrieve a value from the cache by key.
     *
     * @param key the cache key
     * @return an Optional containing the value if present, otherwise empty
     */
    Optional<T> get(String key);

    /**
     * Store a value in the cache by key.
     *
     * @param key the cache key
     * @param value the value to cache
     */
    void put(String key, T value);
}
