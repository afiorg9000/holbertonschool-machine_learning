-- Create a trigger
DELIMITER //
CREATE TRIGGER after_insert_order
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
	UPDATE items
	SET items.quantity = items.quantity - NEW.number
	WHERE items.name = NEW.item_name;
END;
//