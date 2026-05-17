curl -X POST  http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
   "model": "qwen3.5-27b",
   "messages": [
     {
       "role": "system",
       "content": "你是一个AI助手"
     },
     {
       "role": "user",
       "content": [
         {
           "type": "text",
           "text": "直接告诉我这张图片上有什么内容"
         },
         {
           "type": "image_url",
           "image_url": {
             "url": "https://ci.xiaohongshu.com/notes_pre_post/1040g3k831i2blnkin8205npp9u90buk41jc6db0?imageView2/2/w/512/format/jpg"
           }
         }
       ]
      }
   ],
   "stream": true,
   "max_tokens": 200,
   "temperature": 0.9
 }'