package com.example.jwtvalidator.cache.serializer;

/**
 * A generic serializer interface for converting objects to and from a String representation.
 *
 * @param <T> the type of object to serialize
 */
public interface Serializer<T> {
    /**
     * Serialize an object to a String.
     *
     * @param value the object to serialize
     * @return the serialized String
     * @throws Exception if serialization fails
     */
    String serialize(T value) throws Exception;

    /**
     * Deserialize an object from a String.
     *
     * @param data the serialized String
     * @return the deserialized object
     * @throws Exception if deserialization fails
     */
    T deserialize(String data) throws Exception;
} 