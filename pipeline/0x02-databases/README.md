# 0x02 Databases

> A database is an organized collection of structured information, or data, typically stored electronically in a computer system. A database is usually controlled by a database management system (DBMS)

At the end of this project I was able to answer these conceptual questions:

* What’s a relational database
* What’s a none relational database
* What is difference between SQL and NoSQL
* How to create tables with constraints
* How to optimize queries by adding indexes
* What is and how to implement stored procedures and functions in MySQL
* What is and how to implement views in MySQL
* What is and how to implement triggers in MySQL
* What is ACID
* What is a document storage
* What are NoSQL types
* What are benefits of a NoSQL database
* How to query information from a NoSQL database
* How to insert/update/delete information from a NoSQL database
* How to use MongoDB

## Tasks

0. Write a script that creates the database `db_0` in your MySQL server.

    * If the database `db_0` already exists, your script should not fail
    * You are not allowed to use the `SELECT` or `SHOW` statements

1. Write a script that creates a table called `first_table` in the current database in your MySQL server.

    * `first_table` description:
        * `id` INT
        * `name` VARCHAR(256)
    * The database name will be passed as an argument of the `mysql` command
    * If the table `first_table` already exists, your script should not fail
    * You are not allowed to use the `SELECT` or `SHOW` statements

2. Write a script that lists all rows of the table `first_table` in your MySQL server.

    * All fields should be printed
    * The database name will be passed as an argument of the `mysql` command

3. Write a script that inserts a new row in the table `first_table` in your MySQL server.

    * New row:
        * `id` = `89`
        * `name` = `Holberton School`
    * The database name will be passed as an argument of the `mysql` command

4. Write a script that lists all records with a `score >= 10` in the table `second_table` in your MySQL server.

    * Results should display both the score and the name (in this order)
    * Records should be ordered by score (top first)
    * The database name will be passed as an argument of the `mysql` command

5. Write a script that computes the score average of all records in the table `second_table` in your MySQL server.

    * The result column name should be `average`
    * The database name will be passed as an argument of the `mysql` command

6. Import in `hbtn_0c_0` database this table dump: [download](https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-higher-level_programming+/272/temperatures.sql)

    Write a script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending).

7. Import in `hbtn_0c_0` database this table dump: [download](https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-higher-level_programming+/272/temperatures.sql) (same as `Temperatures #0`)

    Write a script that displays the max temperature of each state (ordered by State name).

8. Import the database dump from `hbtn_0d_tvshows` to your MySQL server: [download](https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows.sql)

    Write a script that lists all shows contained in `hbtn_0d_tvshows` that have at least one genre linked.

    * Each record should display: `tv_shows.title` - `tv_show_genres.genre_id`
    * Results must be sorted in ascending order by `tv_shows.title` and `tv_show_genres.genre_id`
    * You can use only one `SELECT` statement
    * The database name will be passed as an argument of the `mysql` command

9. Import the database dump from `hbtn_0d_tvshows` to your MySQL server: [download](https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows.sql)

    Write a script that lists all shows contained in `hbtn_0d_tvshows` without a genre linked.

    * Each record should display: `tv_shows.title` - `tv_show_genres.genre_id`
    * Results must be sorted in ascending order by `tv_shows.title` and `tv_show_genres.genre_id`
    * You can use only one `SELECT` statement
    * The database name will be passed as an argument of the `mysql` command

10. Import the database dump from `hbtn_0d_tvshows` to your MySQL server: [download](https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows.sql)

    Write a script that lists all genres from `hbtn_0d_tvshows` and displays the number of shows linked to each.

    * Each record should display: `<TV Show genre>` - `<Number of shows linked to this genre>`
    * First column must be called `genre`
    * Second column must be called `number_of_shows`
    * Don’t display a genre that doesn’t have any shows linked
    * Results must be sorted in descending order by the number of shows linked
    * You can use only one `SELECT` statement
    * The database name will be passed as an argument of the `mysql` command

11. Import the database `hbtn_0d_tvshows_rate` dump to your MySQL server: [download](https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows_rate.sql)

    Write a script that lists all shows from `hbtn_0d_tvshows_rate` *  by their rating.

    * Each record should display: `tv_shows.title` - `rating sum`
    * Results must be sorted in descending order by the rating
    * You can use only one `SELECT` statement
    * The database name will be passed as an argument of the `mysql` command

12. Import the database dump from `hbtn_0d_tvshows_rate` to your MySQL server: [download](https://s3.eu-west-3.amazonaws.com/hbtn.intranet.project.files/holbertonschool-higher-level_programming+/274/hbtn_0d_tvshows_rate.sql)

    Write a script that lists all genres in the database `hbtn_0d_tvshows_rate` by their rating.

    * Each record should display: `tv_genres.name` - `rating sum`
    * Results must be sorted in descending order by their rating
    * You can use only one `SELECT` statement
    * The database name will be passed as an argument of the `mysql` command

13. Write a SQL script that creates a table `users` following these requirements:

    * With these attributes:
        * `id`, integer, never null, auto increment and primary key
        * `email`, string (255 characters), never null and unique
        * `name`, string (255 characters)
    * If the table already exists, your script should not fail
    * Your script can be executed on any database

14. Write a SQL script that creates a table `users` following these requirements:

    * With these attributes:
        * `id`, integer, never null, auto increment and primary key
        * `email`, string (255 characters), never null and unique
        * `name`, string (255 characters)
        * `country`, enumeration of countries: `US`, `CO` and `TN`, never null (= default will be the first element of the enumeration, here `US`)
    * If the table already exists, your script should not fail
    * Your script can be executed on any database

15. Write a SQL script that ranks country origins of bands, ordered by the number of (non-unique) fans

    **Requirements:**

    * Import this table dump: [metal_bands.sql.zip](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/misc/2020/6/ab2979f058de215f0f2ae5b052739e76d3c02ac5.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230317%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230317T171035Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=c058fdf437f856ebe1385d13e399217d8bbe103123d82018782d6f2ee3595f67)
    * Column names must be: `origin` and `nb_fans`
    * Your script can be executed on any database

16. Write a SQL script that lists all bands with `Glam rock` as their main style, ranked by their longevity

    **Requirements:**

    * Import this table dump: [metal_bands.sql.zip](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/misc/2020/6/ab2979f058de215f0f2ae5b052739e76d3c02ac5.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230317%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230317T171035Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=c058fdf437f856ebe1385d13e399217d8bbe103123d82018782d6f2ee3595f67)
    * Column names must be:
        * `band_name`
        * `lifespan` until 2020 (in years)
    * You should use attributes `formed` and `split` for computing the `lifespan`
    * Your script can be executed on any database

17. Write a SQL script that creates a trigger that decreases the quantity of an item after adding a new order.

    Quantity in the table `items` can be negative.

18. Write a SQL script that creates a trigger that resets the attribute `valid_email` only when the `email` has been changed.

19. Write a SQL script that creates a stored procedure `AddBonus` that adds a new correction for a student.

    **Requirements:**

    * Procedure `AddBonus` is taking 3 inputs (in this order):
        * `user_id`, a `users.id` value (you can assume `user_id` is linked to an existing `users`)
        * `project_name`, a new or already exists `projects` - if no `projects.name` found in the table, you should create it
        * `score`, the score value for the correction

20. Write a SQL script that creates a stored procedure `ComputeAverageScoreForUser` that computes and store the average score for a student.

    **Requirements:**

    * Procedure `ComputeAverageScoreForUser` is taking 1 input:
        * `user_id`, a `users.id` value (you can assume `user_id` is linked to an existing `users`)

21. Write a SQL script that creates a function `SafeDiv` that divides (and returns) the first by the second number or returns 0 if the second number is equal to 0.

    **Requirements:**

    * You must create a function
    * The function `SafeDiv` takes 2 arguments:
        * `a`, INT
        * `b`, INT
    * And returns `a / b` or 0 if `b == 0`

22. Write a script that lists all databases in MongoDB.

23. Write a script that creates or uses the database `my_db`

24. Write a script that inserts a document in the collection `school`:

    * The document must have one attribute `name` with value “Holberton school”
    * The database name will be passed as option of `mongo` command

25. Write a script that lists all documents in the collection `school`:

    * The database name will be passed as option of `mongo` command

26. Write a script that lists all documents with `name="Holberton school"` in the collection `school`:

    * The database name will be passed as option of `mongo` command

27. Write a script that displays the number of documents in the collection `school`:

    * The database name will be passed as option of `mongo` command

28. Write a script that adds a new attribute to a document in the collection `school`:

    * The script should update only document with `name="Holberton school"` (all of them)
    * The update should add the attribute `address` with the value “972 Mission street”
    * The database name will be passed as option of `mongo` command

29. Write a script that deletes all documents with `name="Holberton school"` in the collection `school`:

    * The database name will be passed as option of `mongo` command

30. Write a Python function that lists all documents in a collection:

    * Prototype: `def list_all(mongo_collection):`
    * Return an empty list if no document in the collection
    * `mongo_collection` will be the `pymongo` collection object

31. Write a Python function that inserts a new document in a collection based on `kwargs`:

    * Prototype: `def insert_school(mongo_collection, **kwargs):`
    * `mongo_collection` will be the `pymongo` collection object
    * Returns the new `_id`

32. Write a Python function that changes all topics of a school document based on the name:

    * Prototype: `def update_topics(mongo_collection, name, topics):`
    * `mongo_collection` will be the `pymongo` collection object
    * `name` (string) will be the school name to update
    * `topics` (list of strings) will be the list of topics approached in the school

33. Write a Python function that returns the list of school having a specific topic:

    * Prototype: `def schools_by_topic(mongo_collection, topic):`
    * `mongo_collection` will be the `pymongo` collection object
    * `topic` (string) will be topic searched

34. Write a Python script that provides some stats about Nginx logs stored in MongoDB:

    * Database: `logs`
    * Collection: `nginx`
    * Display:
        * first line: `x logs` where `x` is the number of documents in this collection
        * second line: `Methods:`
        * 5 lines with the number of documents with the `method` = `["GET", "POST", "PUT", "PATCH", "DELETE"]` in this order
        * one line with the number of documents with:
            * `method=GET`
            * `path=/status`
    You can use this dump as data sample: [dump.zip](https://s3.eu-west-3.amazonaws.com/hbtn.intranet/uploads/misc/2020/6/645541f867bb79ae47b7a80922e9a48604a569b9.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4MYA5JM5DUTZGMZG%2F20230317%2Feu-west-3%2Fs3%2Faws4_request&X-Amz-Date=20230317T195310Z&X-Amz-Expires=345600&X-Amz-SignedHeaders=host&X-Amz-Signature=7a32848bee989941928dac0cd9322c73d90cfb7947a25f4b078d0bd1023fb1c8)

