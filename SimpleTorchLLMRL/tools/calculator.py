

from .tool import Tool



class Calculator(Tool):


    def __init__(self) -> None:
        super().__init__(tags = "calculator",
                        description="""This is a calculator tool that does simple addition and multiplicaiton. 
                        you can call it with <calculator> 2+2 </calculator>""",
                        name="Calculator")


    def _execute(self, input:str) -> str | None:
        
        ["(", "2" , "+", "2", ")", "*", "4"]
        #addition
        try:
            add1, add2 = input.split("+", maxsplit=1)
            output = float(add1) + float(add2)
            return str(output)
        except:
            pass
        
        #multiply 
        try:
            add1, add2 = input.split("*", maxsplit=1)
            output =float(add1) * float(add2)
            return str(output)
        except:
            pass
        return None
        
