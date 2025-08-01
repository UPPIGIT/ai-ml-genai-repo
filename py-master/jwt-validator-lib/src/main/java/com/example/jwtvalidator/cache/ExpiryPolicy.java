package com.example.jwtvalidator.cache;

/**
 * A policy interface for determining cache expiry time for a given key and value.
 *
 * @param <T> the type of object being cached
 */
@FunctionalInterface
public interface ExpiryPolicy<T> {
    /**
     * Determine the expiry time (in seconds) for a given key and value.
     *
     * @param key the cache key
     * @param value the value to cache
     * @return the expiry time in seconds
     */
    long getExpirySeconds(String key, T value);
} 