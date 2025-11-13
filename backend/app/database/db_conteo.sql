CREATE TABLE memoria.historial (
  fecha DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3), 
  camara_id INT UNSIGNED NOT NULL,                    
  conteo SMALLINT UNSIGNED NOT NULL,                
  indice_aglomeracion DECIMAL(2,1) NOT NULL,              
  PRIMARY KEY (fecha, camara_id),                       
  INDEX idx_historial_fecha (fecha DESC)
) ENGINE=InnoDB;