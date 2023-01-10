
CREATE DATABASE GameStore;-- Making the database
SHOW DATABASES;
USE GameStore;-- Selecting the Database
-- Creating the Table
CREATE TABLE games (
	ID	int	not null,
    name	varchar(20)	not null,
    price	int,
    sale_amout	int
    );

INSERT INTO games
values
	(1,'MW2',50,20),
    (2,'SC2',40,10),
    (3,'HOTS',0,0),
    (4,'World of warcraft',55,25),
    (5,'NeedForSpeedHeat',80,75),
    (6,'XBOX',899,13),
    (7,'For Honor',75,25);

SELECT *,FORMAT(price * (100 - sale_amout)/100,'###0.') AS 'discount price' FROM games; -- showing the discounted price by the sale_amount rate

DELETE FROM games WHERE ID = 6
