package com.example.jwtvalidator.cache;

import com.auth0.jwk.Jwk;
import java.util.Optional;

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

public class CaffeineCacheService implements CacheService<Jwk> {

    private final Map<String, Jwk> cache;

    public CaffeineCacheService(long cacheExpiry) {
        this.cache = new ConcurrentHashMap<>();
    }

    @Override
    public Optional<Jwk> get(String key) {
        return Optional.ofNullable(cache.get(key));
    }

    @Override
    public void put(String key, Jwk value) {
        cache.put(key, value);
    }
}
