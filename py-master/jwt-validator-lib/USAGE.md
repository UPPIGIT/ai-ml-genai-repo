# JWT Validator Library

This library provides a simple, auto-configurable way to validate JWTs using a JWKS endpoint. It is designed for Spring Boot applications and supports multiple caching strategies to improve performance and reduce latency.

## Features

- **JWT Validation:** Validates JWT signatures against a remote JWKS (JSON Web Key Set) endpoint.
- **Auto-Configuration:** Automatically configures the `JwtValidationService` when the library is included in a Spring Boot project.
- **Flexible Caching:** Supports Redis, Caffeine (in-memory), and no-op caching to store JWKs and minimize network requests.

---

## Getting Started

Follow these steps to integrate the JWT Validator Library into your Spring Boot project.

### 1. Add the Dependency

Include the library as a dependency in your project's `pom.xml`:

```xml
<dependency>
    <groupId>com.example</groupId>
    <artifactId>jwt-validator-lib</artifactId>
    <version>1.0.0-SNAPSHOT</version>
</dependency>
```

### 2. Configure the Library

Add the following properties to your `application.properties` or `application.yml` file to configure the validator:

```properties
# Enable or disable the JWT validation feature. Defaults to true.
jwt.validation.enabled=true

# The full URL of your JWKS endpoint. This is a required property.
jwt.validation.jwks-url=https://your-auth-server/.well-known/jwks.json

# The type of cache to use. Options are: REDIS, CAFFEINE, NONE. Defaults to NONE.
jwt.validation.cache-type=CAFFEINE

# The cache expiration time in seconds. This property is required if caching is enabled.
jwt.validation.cache-expiry=3600
```

### 3. Inject and Use the Service

Inject the `JwtValidationService` into your Spring components (e.g., a controller or another service) and use it to validate JWTs.

```java
import com.example.jwtvalidator.service.JwtValidationService;
import com.example.jwtvalidator.exception.JwtValidationException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class MyController {

    private final JwtValidationService jwtValidationService;

    @Autowired
    public MyController(JwtValidationService jwtValidationService) {
        this.jwtValidationService = jwtValidationService;
    }

    @PostMapping("/validate")
    public String validateToken(@RequestHeader("Authorization") String authorizationHeader) {
        try {
            // The service expects the raw token, so remove the "Bearer " prefix if present.
            String token = authorizationHeader.replace("Bearer ", "");
            jwtValidationService.validateJwtToken(token);
            return "Token is valid!";
        } catch (JwtValidationException e) {
            // The token is invalid.
            return "Token is invalid: " + e.getMessage();
        }
    }
}
```

---

## Caching Explained

Caching is essential for performance, as it prevents the library from fetching the JWKS from the remote URL on every token validation.

-   **`REDIS`**: Caches JWKs in a Redis server. This is ideal for distributed systems where multiple service instances need to share the same cache. This option requires a running Redis instance and the `spring-boot-starter-data-redis` dependency in your project.

-   **`CAFFEINE`**: Caches JWKs in memory using Caffeine. This is a high-performance, in-memory cache perfect for single-instance deployments. It requires no external services. The `caffeine` dependency is included as an optional dependency in this library, so you may need to add it to your `pom.xml` if you use this option.

-   **`NONE`**: Disables caching entirely. The JWKS will be fetched from the `jwksUrl` for every validation request. This is not recommended for production environments.

---

## Cache Configuration Examples

### 1. Redis Cache

Add these properties to your `application.properties`:

```properties
jwt.validation.cache-type=REDIS
jwt.validation.cache-expiry=3600
jwt.validation.jwks-url=https://your-auth-server/.well-known/jwks.json

# Redis connection properties (example)
spring.redis.host=localhost
spring.redis.port=6379
```

> **Note:** You must add the following dependency to your `pom.xml`:
> ```xml
> <dependency>
>   <groupId>org.springframework.boot</groupId>
>   <artifactId>spring-boot-starter-data-redis</artifactId>
> </dependency>
> ```

### 2. Caffeine (In-Memory) Cache

```properties
jwt.validation.cache-type=CAFFEINE
jwt.validation.cache-expiry=3600
jwt.validation.jwks-url=https://your-auth-server/.well-known/jwks.json
```

> **Note:** If you use Caffeine, ensure you have the dependency:
> ```xml
> <dependency>
>   <groupId>com.github.ben-manes.caffeine</groupId>
>   <artifactId>caffeine</artifactId>
> </dependency>
> ```

### 3. No Cache (None)

```properties
jwt.validation.cache-type=NONE
jwt.validation.jwks-url=https://your-auth-server/.well-known/jwks.json
```

> **Note:** With `NONE`, the JWKS will be fetched on every validation request. This is not recommended for production.