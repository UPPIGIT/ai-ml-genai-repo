package com.example.jwtvalidator.config;

import com.example.jwtvalidator.JwtValidationService;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableCaching
@EnableConfigurationProperties(JwtValidationProperties.class)
public class JwtValidationAutoConfiguration {
    @Bean
    @ConditionalOnMissingBean
    public JwtValidationService jwtValidationService(JwtValidationProperties properties) {
        return new JwtValidationService(properties);
    }
} 