-- Create store procedure
DELIMITER //
CREATE PROCEDURE AddBonus (

        IN user_id INT,
        IN project VARCHAR(255),
        IN score INT)
BEGIN
        IF NOT EXISTS (SELECT name FROM projects WHERE name=project) THEN
                INSERT INTO projects (name) VALUES (project);
        END IF;
        INSERT INTO corrections (user_id, project_id, score)
                VALUES (user_id, (SELECT id FROM projects WHERE name = project), score);
END//