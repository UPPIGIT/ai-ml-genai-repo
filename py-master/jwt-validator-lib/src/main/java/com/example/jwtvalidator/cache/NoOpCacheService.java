package com.example.jwtvalidator.cache;

import com.auth0.jwk.Jwk;
import java.util.Optional;

public class NoOpCacheService implements CacheService<Jwk> {
    @Override
    public Optional<Jwk> get(String key) {
        return Optional.empty();
    }

    @Override
    public void put(String key, Jwk value) {
        // Do nothing
    }
}
