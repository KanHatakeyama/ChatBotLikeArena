from google.oauth2.service_account import Credentials
import gspread
scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]


class Client:
    def __init__(self,
                 spreadsheet_url,
                 json_path="env/auth.json"):

        credentials = Credentials.from_service_account_file(
            json_path,
            scopes=scopes,
        )

        gc = gspread.authorize(credentials)

        self.sheet_list = [gc.open_by_url(spreadsheet_url).sheet1,
                           gc.open_by_url(
            spreadsheet_url).get_worksheet_by_id(667188849)]
        self.set_sheet_id(0)

    def set_sheet_id(self, sid):
        self.current_sheet_id = sid
        self.sheet = self.sheet_list[sid]

    def get_q_and_a(self):
        values = self.sheet.get_all_values()
        self.values = values

        questions = []
        answers = []
        instructios = []
        for records in values:
            questions.append(records[0])
            instructios.append("")
            if len(records) > 1:
                answers.append(records[1])
            else:
                answers.append("")

        self.questions = questions
        self.answers = answers
        self.instructios = instructios
        self.values = values
        return questions, answers

    def get_unanswered_question(self):
        self.get_q_and_a()

        for id, (q, a, inst) in enumerate(zip(self.questions, self.answers, self.instructios)):
            if a == "":
                return id+1, q, inst

        return id+1, "", ""

    def answer(self, row_id, answer1, answer2,
               metainfo1="meta1",
               metainfo2="meta2",
               metainfo3="meta3",
               ):
        self.sheet.update(f'B{row_id}', [[answer1]])
        self.sheet.update(f'C{row_id}', [[answer2]])
        self.sheet.update(f'F{row_id}', [[metainfo1]])
        self.sheet.update(f'G{row_id}', [[metainfo2]])
        self.sheet.update(f'H{row_id}', [[metainfo3]])
