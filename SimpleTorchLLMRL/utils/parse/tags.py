




def tag_parse(inside_tags:str, text:str) -> list[str]:
    tag_start_end = []
    search_pos = 0
    start_tag = f"<{inside_tags}>"
    end_tag = f"</{inside_tags}>"
    
    while True:
        start = text.find(start_tag, search_pos)
        if start == -1:
            break
            
        start_inside = start + len(start_tag)
        end = text.find(end_tag, start_inside)
        if end == -1:
            break
            
        tag_start_end.append((start_inside, end))
        search_pos = end + len(end_tag)
    
    return tag_start_end


    
        