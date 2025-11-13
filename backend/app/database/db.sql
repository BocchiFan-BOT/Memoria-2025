CREATE TABLE camaras (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  public_id VARCHAR(32) NOT NULL,              
  name VARCHAR(120) NOT NULL,
  url VARCHAR(600) NOT NULL,                    
  location VARCHAR(120) NULL,
  latitude DECIMAL(9,6) NULL,                  
  longitude DECIMAL(9,6) NULL,                  
  status ENUM('ACTIVE','INACTIVE') DEFAULT 'ACTIVE',
  is_online TINYINT(1) NOT NULL DEFAULT 0,      -- 0/1
  last_heartbeat DATETIME NULL,                 -- último ping/healthcheck
  last_error VARCHAR(500) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY ux_camaras_public_id (public_id),
  UNIQUE KEY ux_camaras_url (url),
  KEY ix_camaras_name_location (name, location),
  KEY ix_camaras_is_online (is_online),
  KEY ix_camaras_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_520_ci;


ALTER TABLE camaras
ADD COLUMN alert_count_threshold INT NULL AFTER longitude,
ADD COLUMN alert_occ_threshold DECIMAL(5,2) NULL AFTER alert_count_threshold;