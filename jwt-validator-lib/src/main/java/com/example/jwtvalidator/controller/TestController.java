package com.example.jwtvalidator.controller;

import com.example.jwtvalidator.exception.JwtValidationException;
import com.example.jwtvalidator.service.JwtValidationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RestController
public class TestController {

    private static final Logger logger = LoggerFactory.getLogger(TestController.class);

    private final JwtValidationService jwtValidationService;

    @Autowired
    public TestController(JwtValidationService jwtValidationService) {
        this.jwtValidationService = jwtValidationService;
    }

    @PostMapping("/validate")
    public ResponseEntity<String> validateToken(@RequestHeader("Authorization") String authorizationHeader) {
        logger.info("Received a request to /validate endpoint.");
        try {
            String token = authorizationHeader.replace("Bearer ", "");
            jwtValidationService.validateJwtToken(token);
            logger.info("Validation successful for the provided token.");
            return ResponseEntity.ok("Token is valid!");
        } catch (JwtValidationException e) {
            logger.warn("Validation failed: {}", e.getMessage());
            return ResponseEntity.status(401).body("Token is invalid: " + e.getMessage());
        } catch (Exception e) {
            logger.error("An unexpected error occurred in the validation controller.", e);
            return ResponseEntity.status(500).body("An unexpected error occurred: " + e.getMessage());
        }
    }
}
