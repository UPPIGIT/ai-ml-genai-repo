# JWT Validator Library — Usage Guide

## Features
- Validates JWT signatures using JWKS endpoint
- Uses Spring Cache abstraction for JWK caching
- Pluggable cache: Caffeine (in-memory), Redis, or any Spring Cache provider
- Spring Boot auto-configuration

---

## 1. Add Dependency

If using the JAR directly:
- Build the library (`mvn clean package`)
- Copy `target/jwt-validator-lib-1.0.0.jar` to your user project’s `libs/` directory

If using Maven:
```xml
<dependency>
    <groupId>com.example</groupId>
    <artifactId>jwt-validator-lib</artifactId>
    <version>1.0.0</version>
</dependency>
```

---

## 2. Configure JWKS and Cache

**application.properties:**
```properties
# Required: JWKS endpoint URL
jwt.validation.jwks-url=https://example.com/.well-known/jwks.json

# Spring Cache options (choose one):

# --- Caffeine (in-memory, recommended for most) ---
spring.cache.type=caffeine
spring.cache.cache-names=jwks
spring.cache.caffeine.spec=expireAfterWrite=3600s

# --- Redis (for distributed cache) ---
# spring.cache.type=redis
# spring.cache.cache-names=jwks
# spring.redis.host=localhost
# spring.redis.port=6379

# --- No cache (not recommended for production) ---
# spring.cache.type=none
```

---

## 3. Use in Your Service

```java
import com.example.jwtvalidator.JwtValidationService;
import com.example.jwtvalidator.JwtValidationException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class MyService {
    @Autowired
    private JwtValidationService jwtValidationService;

    public void processToken(String token) {
        try {
            jwtValidationService.validateJwtToken(token);
            // Token is valid, proceed
        } catch (JwtValidationException e) {
            // Handle invalid token
        }
    }
}
```

---

## 4. Supported Cache Providers

- **Caffeine (default, in-memory, with expiry):**
  - Fast, local, easy to use.
- **Redis:**
  - Distributed, for clustered environments.
- **None:**
  - No caching (fetches JWKS for every validation, not recommended for production).
- **Any Spring Cache-compatible provider:**  
  - EhCache, Hazelcast, etc. (just change Spring Boot config).

---

## 5. Advanced

- **Cache expiry** is controlled by your cache provider’s config (e.g., `expireAfterWrite` for Caffeine).
- **Cache key** is the JWT’s `kid` (Key ID) from the header.
- **Error handling:** Throws `JwtValidationException` for all validation errors.

---

## 6. Example application.properties for Redis

```properties
jwt.validation.jwks-url=https://login.microsoftonline.com/common/discovery/keys
spring.cache.type=redis
spring.cache.cache-names=jwks
spring.redis.host=localhost
spring.redis.port=6379
```

---

## 7. Building from Source

1. Unzip the project.
2. Run:
   ```sh
   mvn clean package
   ```
3. The JAR will be in `target/jwt-validator-lib-1.0.0.jar`.

---

## 8. Using the JAR in Another Project

### A. Local JAR (Quick Test/POC)
1. Copy `jwt-validator-lib-1.0.0.jar` to a `libs/` directory in your user project.
2. In your user project’s `pom.xml`, add:
   ```xml
   <dependency>
       <groupId>com.example</groupId>
       <artifactId>jwt-validator-lib</artifactId>
       <version>1.0.0</version>
       <scope>system</scope>
       <systemPath>${project.basedir}/libs/jwt-validator-lib-1.0.0.jar</systemPath>
   </dependency>
   ```
   > Note: `<scope>system</scope>` is fine for local testing, but not recommended for production.

### B. Install to Local Maven Repo (Recommended for Teams)
1. Run:
   ```sh
   mvn install:install-file -Dfile=target/jwt-validator-lib-1.0.0.jar -DgroupId=com.example -DartifactId=jwt-validator-lib -Dversion=1.0.0 -Dpackaging=jar
   ```
2. In your user project’s `pom.xml`, add:
   ```xml
   <dependency>
       <groupId>com.example</groupId>
       <artifactId>jwt-validator-lib</artifactId>
       <version>1.0.0</version>
   </dependency>
   ```

### C. Publish to a Maven Repository
- If you have a private Maven/Nexus/Artifactory repo, publish the JAR there and use the standard dependency declaration.

### D. Add Required Cache Provider Dependencies
Add the cache provider you want to your user project:
```xml
<!-- For Caffeine (in-memory) -->
<dependency>
    <groupId>com.github.ben-manes.caffeine</groupId>
    <artifactId>caffeine</artifactId>
</dependency>
<!-- For Redis (if using Redis cache) -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>
```

---

## 9. License

MIT (or your preferred license)

---

## 10. Support

For issues or feature requests, contact the maintainer. 