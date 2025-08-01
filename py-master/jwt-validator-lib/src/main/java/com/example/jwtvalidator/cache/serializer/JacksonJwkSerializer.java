package com.example.jwtvalidator.cache.serializer;

import com.auth0.jwk.Jwk;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.security.interfaces.RSAPublicKey;
import java.util.HashMap;
import java.util.Map;

public class JacksonJwkSerializer implements Serializer<Jwk> {
    private final ObjectMapper objectMapper;

    public JacksonJwkSerializer(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @Override
    public String serialize(Jwk jwk) throws Exception {
        Map<String, Object> jwkMap = new HashMap<>();
        jwkMap.put("kty", jwk.getType());
        jwkMap.put("kid", jwk.getId());
        jwkMap.put("alg", jwk.getAlgorithm());
        jwkMap.put("use", jwk.getUsage());
        if (jwk.getPublicKey() instanceof RSAPublicKey) {
            RSAPublicKey rsaKey = (RSAPublicKey) jwk.getPublicKey();
            jwkMap.put("n", rsaKey.getModulus().toString(16));
            jwkMap.put("e", rsaKey.getPublicExponent().toString(16));
        }
        jwkMap.putAll(jwk.getAdditionalAttributes());
        return objectMapper.writeValueAsString(jwkMap);
    }

    @Override
    public Jwk deserialize(String data) throws Exception {
        Map<String, Object> jwkMap = objectMapper.readValue(data, new TypeReference<Map<String, Object>>() {});
        return Jwk.fromValues(jwkMap);
    }
} 