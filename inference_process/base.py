class Inference_skeleton():
    def __init__(self, args:dict) -> None:
        super(Inference_skeleton, self).__init__()

        ## Validation Check
        args = self.convert_to_object(args)
        self.validation_check(args)

    ## 반드시 구현해야 함.
    def get_output(self, input: list) -> tuple:
        raise NotImplementedError

    ## 유효성 Check function
    def validation_check(self, args) -> bool:
        return True

    ## dict로 받은 파일을 object로 변경 function
    def convert_to_object(self, args:dict):
        class Struct:
            def __init__(self, **entries):
                self.__dict__.update(entries)
        return Struct(**args)
