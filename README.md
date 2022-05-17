# python-flask-example

Code example used in [Building Python applications](https://cloud.google.com/build/docs/building/build-python). For instructions on running this code sample, see the documentation.

```SQL
CREATE TABLE IF NOT EXISTS votes
( vote_id SERIAL NOT NULL, time_cast timestamp NOT NULL,
candidate VARCHAR(6) NOT NULL, PRIMARY KEY (vote_id) );

CREATE TABLE IF NOT EXISTS totals
( total_id SERIAL NOT NULL, candidate VARCHAR(6) NOT NULL,
num_votes INT DEFAULT 0, PRIMARY KEY (total_id) );

INSERT INTO totals (candidate, num_votes) VALUES ('TABS', 0);
INSERT INTO totals (candidate, num_votes) VALUES ('SPACES', 0);

CREATE TABLE IF NOT EXISTS mle
( key VARCHAR(30) NOT NULL, value VARCHAR(50) NOT NULL,
PRIMARY KEY (key) );

insert into mle(key,value) values ('crawling',0);
insert into mle(key,value) values ('lastCrawledDate',0);
insert into mle(key,value) values ('numOfDaysToTrain',30);


CREATE TABLE careers (
  uuid varchar(45) DEFAULT NULL,
  title varchar(1000) DEFAULT NULL,
  description varchar(5000) DEFAULT NULL,
  minimumYearsExperience int DEFAULT NULL,
  skills varchar(1000) DEFAULT NULL,
  numberOfVacancies int DEFAULT NULL,
  categories varchar(1000) DEFAULT NULL,
  employmentTypes varchar(450) DEFAULT NULL,
  positionLevels varchar(1000) DEFAULT NULL,
  totalNumberOfView int DEFAULT NULL,
  totalNumberJobApplication int DEFAULT NULL,
  originalPostingDate date DEFAULT NULL,
  expiryDate date DEFAULT NULL,
  links varchar(1000) DEFAULT NULL,
  postedCompany varchar(1000) DEFAULT NULL,
  minsalary int DEFAULT NULL,
  maxsalary int DEFAULT NULL,
  avgsalary int DEFAULT NULL,
  remarks varchar(1000) DEFAULT NULL,
  included varchar(20) DEFAULT NULL,
  crawldate timestamp DEFAULT NULL,
  PRIMARY KEY (uuid));

  UPDATE mle SET value = '0' WHERE key='crawling';
  SELECT value FROM mle
            where key='crawling';
```
