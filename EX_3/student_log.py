import re
from collections import defaultdict

# -----------------------------
# Student Class
# -----------------------------
class Student:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name
        self.activities = []  # list of (activity, date, time)

    def add_activity(self, activity, date, time):
        self.activities.append((activity, date, time))

    def activity_summary(self):
        logins = sum(1 for a in self.activities if a[0] == "LOGIN")
        submissions = sum(1 for a in self.activities if a[0] == "SUBMIT_ASSIGNMENT")
        return logins, submissions


# -----------------------------
# Generator Function
# -----------------------------
def read_log_file(filename):
    """
    Reads the log file line by line and yields valid records only
    """
    student_id_pattern = re.compile(r"^S\d+$")
    valid_activities = {"LOGIN", "LOGOUT", "SUBMIT_ASSIGNMENT"}

    with open(filename, "r") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) != 5:
                    raise ValueError("Incorrect number of fields")

                student_id, name, activity, date, time = parts

                if not student_id_pattern.match(student_id):
                    raise ValueError("Invalid student ID")

                if activity not in valid_activities:
                    raise ValueError("Invalid activity type")

                yield student_id, name, activity, date, time

            except Exception as e:
                print(f"Skipping invalid line {line_number}: {e}")


# -----------------------------
# Main Processing Logic
# -----------------------------
def process_logs(input_file, output_file):
    students = {}
    daily_stats = defaultdict(lambda: {"LOGIN": 0, "SUBMIT_ASSIGNMENT": 0})
    abnormal_behavior = defaultdict(int)

    for student_id, name, activity, date, time in read_log_file(input_file):

        if student_id not in students:
            students[student_id] = Student(student_id, name)

        students[student_id].add_activity(activity, date, time)

        # Daily statistics
        if activity in daily_stats[date]:
            daily_stats[date][activity] += 1

        # Abnormal behavior detection
        if activity == "LOGIN":
            abnormal_behavior[student_id] += 1
        elif activity == "LOGOUT":
            abnormal_behavior[student_id] = max(0, abnormal_behavior[student_id] - 1)

    # -----------------------------
    # Generate Report
    # -----------------------------
    report_lines = []
    report_lines.append("STUDENT ACTIVITY REPORT\n")

    for student in students.values():
        logins, submissions = student.activity_summary()
        report_lines.append(
            f"{student.student_id} | {student.name} | "
            f"Logins: {logins} | Submissions: {submissions}"
        )

    report_lines.append("\nABNORMAL BEHAVIOR (Multiple logins without logout):")
    for sid, count in abnormal_behavior.items():
        if count > 1:
            report_lines.append(f"{sid} has {count} active logins")

    report_lines.append("\nDAILY ACTIVITY STATISTICS:")
    for date, stats in daily_stats.items():
        report_lines.append(
            f"{date} -> Logins: {stats['LOGIN']}, Submissions: {stats['SUBMIT_ASSIGNMENT']}"
        )

    # -----------------------------
    # Output to Console
    # -----------------------------
    print("\n".join(report_lines))

    # -----------------------------
    # Write to Output File
    # -----------------------------
    with open(output_file, "w") as file:
        file.write("\n".join(report_lines))


# -----------------------------
# Program Execution
# -----------------------------
if __name__ == "__main__":
    input_log_file = "student_log.txt"
    output_report_file = "student_report.txt"
    process_logs(input_log_file, output_report_file)
