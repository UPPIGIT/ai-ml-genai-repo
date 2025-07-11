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

public class JwtValidationService {
    private static final Logger logger = LoggerFactory.getLogger(JwtValidationService.class);

    private final JwtValidationProperties properties;

    public JwtValidationService(JwtValidationProperties properties) {
        this.properties = properties;
    }

    @Cacheable(value = "jwks", key = "#keyId")
    public Jwk fetchJwk(String keyId) throws JwkException {
        logger.info("Fetching JWK from endpoint for keyId: {}", keyId);
        UrlJwkProvider provider = new UrlJwkProvider(properties.getJwksUrl());
        return provider.get(keyId);
    }

    public void validateJwtToken(String token) throws JwtValidationException {
        try {
            DecodedJWT jwt = JWT.decode(token.replace("Bearer ", ""));
            String keyId = jwt.getKeyId();
            if (keyId == null || keyId.isEmpty()) {
                logger.error("KeyId is null or empty.");
                throw new JwtValidationException("KeyId is null or empty.");
            }

            Jwk jwk = fetchJwk(keyId);

            Algorithm algorithm;
            if ("RSA".equals(jwk.getType())) {
                algorithm = Algorithm.RSA256((RSAPublicKey) jwk.getPublicKey(), null);
            } else if ("EC".equals(jwk.getType())) {
                algorithm = Algorithm.ECDSA256((ECPublicKey) jwk.getPublicKey(), null);
            } else {
                throw new JwtValidationException("Unsupported key type: " + jwk.getType());
            }

            algorithm.verify(jwt);
            logger.info("JWT token signature verified.");
        } catch (JwkException e) {
            logger.error("Failed to fetch JWK: {}", e.getMessage());
            throw new JwtValidationException("Failed to fetch JWK: " + e.getMessage(), e);
        } catch (Exception e) {
            logger.error("JWT validation failed: {}", e.getMessage());
            throw new JwtValidationException("JWT validation failed: " + e.getMessage(), e);
        }
    }
} 