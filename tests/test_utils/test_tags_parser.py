


from SimpleTorchLLMRL.utils.parse.tags import tag_parse


def test_tag_parse_no_tags():
    tag = "calculator"
    text = "kajsdln asdlfndlsaknf laskdncv lkadsmcv adsz"
    assert tag_parse(inside_tags=tag,text=text) == []


    

def test_tag_parse_one_tag():
    tag = "calculator"
    text = "kajsdln <calculator>bob</calculator>asdlfndlsaknf laskdncv lkadsmcv adsz"
    assert len(tag_parse(inside_tags=tag,text=text)) == 1
    start, end = tag_parse(inside_tags=tag,text=text)[0]
    assert text[start:end] == "bob"

    
def test_tag_parse_open_tag():
    tag = "calculator"
    text = "kajsdln <calculator>bobasdlfndlsaknf laskdncv lkadsmcv adsz"
    assert len(tag_parse(inside_tags=tag,text=text)) == 0


def test_tag_parse_open_trailing_tag():
    tag = "calculator"
    text = "kajsdln bobasdlfndlsaknf laskdncv lkadsmcv adsz</calculator>"
    assert len(tag_parse(inside_tags=tag,text=text)) == 0


def test_tag_parse_multi_tag():
    tag = "my super cool tag"
    text = "<my super cool tag>John</my super cool tag>kajsdln <my super cool tag>bob</my super cool tag>asdlfndlsaknf laskdncv lkadsmcv adsz"
    assert len(tag_parse(inside_tags=tag,text=text)) == 2
    start, end = tag_parse(inside_tags=tag,text=text)[0]
    assert text[start:end] == "John"
    start, end = tag_parse(inside_tags=tag,text=text)[1]
    assert text[start:end] == "bob"

    