package com.example.jwtvalidator.config;

import com.example.jwtvalidator.service.JwtValidationService;
import com.example.jwtvalidator.cache.CacheService;
import com.example.jwtvalidator.cache.CaffeineCacheService;
import com.example.jwtvalidator.cache.NoOpCacheService;
import com.example.jwtvalidator.cache.RedisCacheService;
import com.example.jwtvalidator.cache.ExpiryPolicy;
import com.example.jwtvalidator.cache.serializer.Serializer;
import com.example.jwtvalidator.cache.serializer.JacksonJwkSerializer;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.core.StringRedisTemplate;
import com.auth0.jwk.Jwk;


@Configuration
@ComponentScan("com.example.jwtvalidator")
@EnableConfigurationProperties(JwtValidationProperties.class)
@ConditionalOnProperty(name = "jwt.validation.enabled", havingValue = "true", matchIfMissing = true)
public class JwtValidationAutoConfiguration {

        @Bean
    public JwtValidationService jwtValidationService(JwtValidationProperties properties, CacheService<Jwk> cacheService, ObjectMapper objectMapper) {
        return new JwtValidationService(cacheService, properties.getJwksUrl(), objectMapper, properties.getCacheType());
    }

    @Bean
    @ConditionalOnProperty(name = "jwt.validation.cache-type", havingValue = "REDIS")
    @ConditionalOnClass(StringRedisTemplate.class)
    public CacheService<Jwk> redisCacheService(StringRedisTemplate redisTemplate, ObjectMapper objectMapper, JwtValidationProperties properties) {
        if (properties.getCacheExpiry() <= 0) {
            throw new IllegalArgumentException("jwt.validation.cache-expiry must be a positive value when Redis caching is enabled.");
        }
        Serializer<Jwk> serializer = new JacksonJwkSerializer(objectMapper);
        ExpiryPolicy<Jwk> expiryPolicy = (key, value) -> properties.getCacheExpiry();
        return new RedisCacheService<Jwk>(redisTemplate, serializer, expiryPolicy, properties.getRedisKeyPrefix());
    }

    @Bean
    @ConditionalOnProperty(name = "jwt.validation.cache-type", havingValue = "CAFFEINE")
    @ConditionalOnClass(name = "com.github.benmanes.caffeine.cache.Caffeine")
    public CacheService<Jwk> caffeineCacheService(JwtValidationProperties properties) {
        if (properties.getCacheExpiry() <= 0) {
            throw new IllegalArgumentException("jwt.validation.cache-expiry must be a positive value when Caffeine caching is enabled.");
        }
        return new CaffeineCacheService(properties.getCacheExpiry());
    }

    @Bean
    @ConditionalOnProperty(name = "jwt.validation.cache-type", havingValue = "NONE", matchIfMissing = true)
    public CacheService<Jwk> noOpCacheService() {
        return new NoOpCacheService();
    }
}
