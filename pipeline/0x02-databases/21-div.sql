-- Create function to divide two columns
DELIMITER //
CREATE FUNCTION SafeDiv (
        a INT,
        b INT
)
RETURNS FLOAT
DETERMINISTIC
BEGIN
        DECLARE divide FLOAT;
        IF (b = 0) THEN
                SET divide = 0;
        ELSE
                SET divide = a / b;
        END IF;
        RETURN (divide);
END //