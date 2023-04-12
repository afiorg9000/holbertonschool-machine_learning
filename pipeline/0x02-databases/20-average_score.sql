-- PROCEDURE AVERAGE SCORE
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser (
        IN user_id_corrections INT
)
BEGIN
        UPDATE users
        SET average_score = (SELECT
                                    AVG(score)
                             FROM
                                    corrections
                             WHERE
                                     user_id=user_id_corrections)
        WHERE id=user_id_corrections;
END //