SELECT TOP 50000 Title, Body, Tags, 
               Id, Score, ViewCount, 
               FavoriteCount, AnswerCount
FROM Posts 
WHERE ViewCount > 10 
--AND FavoriteCount > 10
AND Score > 10
AND AnswerCount > 0
AND LEN(Tags) - LEN(REPLACE(Tags, '<','')) >= 5