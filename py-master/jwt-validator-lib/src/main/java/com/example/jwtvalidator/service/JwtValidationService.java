package com.example.jwtvalidator.service;

import com.auth0.jwk.Jwk;
import com.auth0.jwk.JwkException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.example.jwtvalidator.exception.JwtValidationException;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.auth0.jwt.JWT;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.exceptions.JWTDecodeException;
import com.auth0.jwt.interfaces.DecodedJWT;
import com.example.jwtvalidator.cache.CacheService;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.security.interfaces.RSAPublicKey;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import com.example.jwtvalidator.config.JwtValidationProperties;

public class JwtValidationService {

    private static final Logger logger = LoggerFactory.getLogger(JwtValidationService.class);

    private final CacheService<Jwk> cacheService;
    private final String jwksUrl;
    private final ObjectMapper objectMapper;
    private final JwtValidationProperties.CacheType cacheType;

    public JwtValidationService(CacheService<Jwk> cacheService, String jwksUrl, ObjectMapper objectMapper, JwtValidationProperties.CacheType cacheType) {
        this.cacheService = cacheService;
        this.jwksUrl = jwksUrl;
        this.objectMapper = objectMapper;
        this.cacheType = cacheType;
    }

    public void validateJwtToken(String token) throws JwtValidationException {
        logger.info("Starting JWT validation.");
        String keyId = "unknown";
        try {
            String jwtToken = token.replace("Bearer ", "");
            DecodedJWT jwt = JWT.decode(jwtToken);
            keyId = jwt.getKeyId();
            logger.debug("Token decoded successfully. Key ID (kid): {}", keyId);

            if (keyId == null || keyId.isEmpty()) {
                logger.error("JWT validation failed: Key ID (kid) is missing from the token header.");
                throw new JwtValidationException("KeyId is null.");
            }

            Jwk jwk = getJwk(keyId);

            if (!(jwk.getPublicKey() instanceof RSAPublicKey)) {
                throw new JwtValidationException("PublicKey is not an RSAPublicKey");
            }
            RSAPublicKey publicKey = (RSAPublicKey) jwk.getPublicKey();

            Algorithm algorithm;
            switch (jwt.getAlgorithm()) {
                case "RS256":
                    algorithm = Algorithm.RSA256(publicKey, null);
                    break;
                case "RS384":
                    algorithm = Algorithm.RSA384(publicKey, null);
                    break;
                case "RS512":
                    algorithm = Algorithm.RSA512(publicKey, null);
                    break;
                default:
                    throw new JwtValidationException("Unsupported signature algorithm: " + jwt.getAlgorithm());
            }

            algorithm.verify(jwt);
            logger.info("JWT token signature verified successfully.");

        } catch (JWTDecodeException e) {
            logger.error("Failed to decode JWT token.", e);
            throw new JwtValidationException("Failed to decode JWT token", e);
        } catch (JwkException e) {
            // At this point, keyId is known and not sensitive.
            logger.error("Invalid signature. Could not retrieve or use key with kid: {}", keyId, e);
            throw new JwtValidationException("Invalid signature", e);
        } catch (Exception e) {
            logger.error("An unexpected error occurred during JWT validation.", e);
            throw new JwtValidationException("Error validating JWT token", e);
        }
    }

    private Jwk getJwk(String keyId) throws JwkException {
        logger.debug("Attempting to retrieve JWK for key ID: {}", keyId);
        Optional<Jwk> cachedJwk = cacheService.get(keyId);

        if (cachedJwk.isPresent()) {
            logger.info("Cache hit for JWK with kid: {}", keyId);
            return cachedJwk.get();
        }

        if (cacheType == JwtValidationProperties.CacheType.NONE) {
            logger.info("Caching is disabled. Fetching JWKS from URL: {}", jwksUrl);
        } else {
            logger.info("Cache miss for JWK with kid: {}. Using {} cache. Fetching JWKS from URL: {}", keyId, cacheType, jwksUrl);
        }
        List<Jwk> jwks = fetchJwksFromUrl();
        logger.debug("Fetched {} keys from URL. Caching them now.", jwks.size());
        if (jwks != null && !jwks.isEmpty()) {
            for (Jwk jwk : jwks) {
                String kid = jwk.getId();
                if (kid != null && !kid.isEmpty()) {
                    cacheService.put(kid, jwk);
                }
            }
        } else {
            logger.warn("Fetched JWKS is empty. Not caching.");
        }
        return findJwkInList(jwks, keyId);
    }

    private Jwk findJwkInList(List<Jwk> jwks, String keyId) throws JwkException {
        return jwks.stream()
                .filter(jwk -> keyId.equals(jwk.getId()))
                .findFirst()
                .orElseThrow(() -> new JwkException("JWK with keyId '" + keyId + "' not found in the provided list."));
    }

    private List<Jwk> fetchJwksFromUrl() throws JwkException {
        try {
            URL url = new URI(jwksUrl).toURL();
            try (InputStream inputStream = url.openStream()) {
                Map<String, Object> jsonMap = objectMapper.readValue(inputStream,
                        new TypeReference<Map<String, Object>>() {
                        });

                @SuppressWarnings("unchecked")
                List<Map<String, Object>> keys = (List<Map<String, Object>>) jsonMap.get("keys");

                if (keys == null || keys.isEmpty()) {
                    throw new JwkException("No keys found in JWKS from " + jwksUrl);
                }

                return keys.stream()
                        .map(map -> Jwk.fromValues(map))
                        .collect(Collectors.toList());
            }
        } catch (IOException | ClassCastException | URISyntaxException e) {
            throw new JwkException("Failed to fetch or parse JWKS from URL: " + jwksUrl, e);
        }
    }
}
