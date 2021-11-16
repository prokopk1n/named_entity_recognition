import date_regex


DATE_MONTH_YEAR_REGEX = r"(?:(?:январь|февраль|март|апрель|июнь|июль|август|сентябрь|октябрь|ноябрь|декабрь)[ \n][^ " \
                        r"\n.,:!?]+[ \n]года)"


DATE_REGEXP = rf"{date_regex.DATE_REGEXP}|{DATE_MONTH_YEAR_REGEX}"
EXTRA1_REGEX = r"(?:(?:осенью|весной|летом|зимой)|(?:к (?:весне|лету|зиме|осени)))[ \n]\d+[ \n]года"
EXTRA_REGEX = r"(?:в середине|в начале|в конце|начала|конца|середины|на рубеже|рубежа)(?:(?:[ \n](?:\d+|[XIV]+)[ \n](?:веков|век[аеу]?))|[ \n]\d+-х годов)"
DATE_REGEXP_FULL = rf"(?:{DATE_REGEXP})|(?:{EXTRA_REGEX})|(?:{EXTRA1_REGEX})"
