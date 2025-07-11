package com.example.jwtvalidator;

import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.interfaces.DecodedJWT;
import com.auth0.jwk.*;
import com.example.jwtvalidator.config.JwtValidationProperties;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.cache.annotation.Cacheable;

import java.security.interfaces.ECPublicKey;
import java.security.interfaces.RSAPublicKey;

/**
 * Service for validating JWT tokens using a JWKS endpoint.
 * Uses Spring Cache for JWK caching.
 */
public class JwtValidationService {
    private static final Logger logger = LoggerFactory.getLogger(JwtValidationService.class);

    private final JwtValidationProperties properties;

    public JwtValidationService(JwtValidationProperties properties) {
        this.properties = properties;
    }

    /**
     * Fetches the JWK for the given keyId, using Spring Cache.
     * If not cached, fetches from the JWKS endpoint and caches the result.
     *
     * @param keyId The Key ID from the JWT header.
     * @return The JWK for the keyId.
     * @throws JwkException if the key cannot be fetched.
     */
    @Cacheable(value = "jwks", key = "#keyId")
    public Jwk fetchJwk(String keyId) throws JwkException {
        logger.info("Fetching JWK from endpoint for keyId: {}", keyId);
        UrlJwkProvider provider = new UrlJwkProvider(properties.getJwksUrl());
        return provider.get(keyId);
    }

    /**
     * Validates the JWT token's signature using the JWKS endpoint.
     * @param token The JWT token (with or without 'Bearer ' prefix).
     * @return true if valid, false if invalid.
     */
    public boolean validateJwtToken(String token) {
        try {
            // Remove 'Bearer ' prefix if present
            String cleanToken = token.replace("Bearer ", "");
            DecodedJWT jwt = JWT.decode(cleanToken);

            String keyId = jwt.getKeyId();
            if (keyId == null || keyId.isEmpty()) {
                logger.error("KeyId is null or empty.");
                return false;
            }

            // Fetch JWK (cached)
            Jwk jwk = fetchJwk(keyId);

            // Dynamically select algorithm based on key type
            Algorithm algorithm;
            if ("RSA".equals(jwk.getType())) {
                algorithm = Algorithm.RSA256((RSAPublicKey) jwk.getPublicKey(), null);
            } else if ("EC".equals(jwk.getType())) {
                algorithm = Algorithm.ECDSA256((ECPublicKey) jwk.getPublicKey(), null);
            } else {
                logger.error("Unsupported key type: {}", jwk.getType());
                return false;
            }

            // Verify the JWT signature
            algorithm.verify(jwt);
            logger.info("JWT token signature verified for keyId: {}", keyId);
            return true;
        } catch (Exception e) {
            logger.error("JWT validation failed: {}", e.getMessage());
            return false;
        }
    }
} 