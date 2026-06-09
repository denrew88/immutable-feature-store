package fs.io.common;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/**
 * 여러 프로세스가 같은 파일을 동시에 고치지 못하게 막는 파일 기반 lock입니다.
 *
 * <p>이 lock은 Java의 {@code FileLock}이나 OS별 전용 lock API를 직접 쓰지 않습니다.
 * 대신 보호하려는 파일 옆에 별도의 {@code .lock} 파일을 두고, 그 파일을 먼저 만든
 * 프로세스만 작업을 진행한다는 규칙을 사용합니다. 예를 들어 {@code state.json}을
 * 보호하려면 {@code state.json.lock}을 먼저 만들고, 작업이 끝나면 그 lock 파일을
 * 지웁니다.</p>
 *
 * <p>핵심은 {@code createNewFile()}의 보장입니다. 이 메서드는 lock 파일이 아직 없을
 * 때만 새 파일을 만들고 성공을 반환합니다. 이미 같은 이름의 파일이 있으면 새로 만들지
 * 않고 실패를 반환합니다. 파일이 있는지 확인하는 일과 새 파일을 만드는 일이 중간에
 * 끊기지 않고 처리되므로, 여러 프로세스가 거의 동시에 호출하더라도 성공하는 쪽은
 * 하나뿐입니다. 따라서 이 코드는 "lock 파일 만들기에 성공한 프로세스만 작업을 진행한다"는
 * 규칙으로 동시 write를 막습니다.</p>
 *
 * <p>lock 파일 안에는 매번 새로 뽑은 UUID token과 pid, host, thread 이름을 적어 둡니다.
 * token은 "이 lock 파일이 아직 내가 만든 파일이 맞는지" 확인하기 위한 표식입니다.
 * release 시점에 token이 다르면, 그 사이 다른 프로세스가 같은 이름의 lock을 새로 잡은
 * 상황일 수 있으므로 절대 삭제하지 않습니다. pid와 host는 timeout이나 현장 장애가 났을 때
 * 누가 lock을 잡았는지 추적하기 위한 디버깅 정보입니다.</p>
 *
 * <p>Windows에서는 백신, IDE, 파일 인덱서가 lock 파일을 잠깐 열고 있어서 삭제가 실패하는
 * 경우가 있습니다. 그래서 lock 해제와 오래된 lock 정리는 한 번 실패했다고 바로 포기하지
 * 않고 짧게 몇 번 재시도합니다.</p>
 */
public final class FilePathLock implements AutoCloseable {
    private static final long[] RELEASE_RETRY_DELAYS_MILLIS = new long[] {25L, 50L, 100L, 200L, 400L};
    private static final String STALE_MILLIS_PROPERTY = "fs.fileLockStaleMillis";
    private static final String STALE_MILLIS_ENV = "FS_FILE_LOCK_STALE_MILLIS";
    private static final String CURRENT_PID = resolveCurrentPid();
    private static final String CURRENT_HOST = resolveCurrentHost();

    private final File path;
    private final long timeoutMillis;
    private final long staleLockMillis;
    private final String token;
    private final String ownerText;
    private boolean acquired;

    public FilePathLock(String path, long timeoutMillis) {
        this(path, timeoutMillis, defaultStaleLockMillis());
    }

    /**
     * lock 객체를 생성합니다. 이 시점에는 아직 lock을 잡지 않습니다.
     *
     * <p>생성자는 경로, timeout, token 같은 준비 정보만 만듭니다. 실제로 lock 파일을
     * 만들고 소유권을 얻는 일은 {@link #acquire()}에서 합니다. 이렇게 나누어 두면 객체를
     * 만든 뒤 try/finally 구조 안에서 원하는 시점에 lock을 잡을 수 있습니다.</p>
     *
     * <p>{@code staleLockMillis}는 "프로세스가 죽어서 lock 파일만 남은 것처럼 보일 때"를
     * 처리하기 위한 선택 기능입니다. 기본값은 0이라 자동 회수를 하지 않습니다. 운영 환경에서
     * stale 회수를 켜려면, 이 값보다 오래된 lock만 후보로 보고 pid/host까지 확인한 뒤에만
     * 지웁니다.</p>
     *
     * @param path lock 파일 경로입니다. 보호하려는 파일과 별도의 {@code .lock} 경로를 넘깁니다.
     * @param timeoutMillis lock 획득을 기다릴 최대 시간입니다.
     * @param staleLockMillis 이 시간보다 오래된 lock만 stale 후보로 봅니다. 0이면 자동 회수를 하지 않습니다.
     */
    public FilePathLock(String path, long timeoutMillis, long staleLockMillis) {
        this.path = new File(path).getAbsoluteFile();
        this.timeoutMillis = timeoutMillis;
        this.staleLockMillis = Math.max(0L, staleLockMillis);
        this.token = UUID.randomUUID().toString();
        this.ownerText = buildOwnerText(this.path, this.token);
    }

    /**
     * lock 파일 만들기를 시도하고, 성공하면 이 객체가 lock을 소유합니다.
     *
     * <p>흐름은 단순합니다. 먼저 lock 파일을 만들어 봅니다. 만들기에 성공하면 lock을
     * 얻은 것이므로 바로 반환합니다. 이미 lock 파일이 있으면 다른 프로세스가 작업 중이라고
     * 보고 잠깐 기다린 뒤 다시 시도합니다. 이 과정을 {@code timeoutMillis}까지 반복합니다.</p>
     *
     * <p>stale lock 회수가 켜져 있으면, 기다리는 동안 {@link #maybeDeleteStaleLock()}도
     * 호출합니다. 다만 이 코드는 "아마 죽은 프로세스의 lock일 것이다" 정도로는 지우지
     * 않습니다. 같은 host인지, pid가 정말 살아 있지 않은지, 삭제 직전에도 token이 그대로인지
     * 확인합니다. 하나라도 애매하면 lock을 지우지 않고 계속 기다립니다. 잘못 지워서 동시에
     * 쓰는 상황을 만드는 것보다 timeout으로 실패하는 편이 안전하기 때문입니다.</p>
     *
     * @throws IOException lock directory 생성, lock 파일 생성, timeout, interrupt 등으로 lock을 얻지 못한 경우
     */
    public void acquire() throws IOException {
        File parent = path.getParentFile();
        if (parent != null && !parent.exists() && !parent.mkdirs()) {
            throw new IOException("failed to create lock directory: " + parent.getAbsolutePath());
        }
        long deadline = System.currentTimeMillis() + timeoutMillis;
        IOException lastAcquireError = null;
        while (true) {
            try {
                if (tryCreateOwnerFile()) {
                    acquired = true;
                    return;
                }
            } catch (IOException e) {
                lastAcquireError = e;
            }
            if (maybeDeleteStaleLock()) {
                continue;
            }
            if (System.currentTimeMillis() >= deadline) {
                IOException timeout = new IOException(
                        "timed out acquiring file lock: " + path.getAbsolutePath()
                                + " existing_lock={" + describeExistingLock() + "}");
                if (lastAcquireError != null) {
                    timeout.addSuppressed(lastAcquireError);
                }
                throw timeout;
            }
            sleepMillis(50L, "acquiring file lock: " + path.getAbsolutePath());
        }
    }

    /**
     * 현재 객체가 소유한 lock 파일을 지워서 다른 프로세스가 작업할 수 있게 합니다.
     *
     * <p>해제할 때는 파일 이름만 보고 지우지 않습니다. 먼저 lock 파일 본문을 다시 읽고,
     * 그 안의 token이 이 객체가 만든 token과 같은지 확인합니다. token이 다르면 같은 경로의
     * lock이라도 다른 프로세스가 새로 잡은 lock일 수 있으므로 삭제하지 않고 실패시킵니다.
     * 이 검사가 없으면, 늦게 release하는 프로세스가 다른 프로세스의 lock을 지워 버릴 수
     * 있습니다.</p>
     *
     * <p>token이 맞는데도 삭제가 실패할 수 있습니다. 특히 Windows에서는 백신이나 IDE가
     * 파일을 짧게 잡고 있어서 {@code delete()}가 실패하는 일이 있습니다. 이런 경우를 위해
     * 짧은 간격으로 몇 번 다시 시도합니다.</p>
     *
     * @throws IOException owner token이 맞지 않거나, retry 후에도 lock 파일을 삭제하지 못한 경우
     */
    public void release() throws IOException {
        if (!acquired) {
            return;
        }

        IOException last = null;
        for (int attempt = 0; attempt <= RELEASE_RETRY_DELAYS_MILLIS.length; attempt++) {
            try {
                releaseOnce();
                acquired = false;
                return;
            } catch (OwnerMismatchException e) {
                throw e;
            } catch (IOException e) {
                last = e;
            }
            if (attempt < RELEASE_RETRY_DELAYS_MILLIS.length) {
                sleepMillis(
                        RELEASE_RETRY_DELAYS_MILLIS[attempt],
                        "releasing file lock: " + path.getAbsolutePath());
            }
        }
        throw new IOException("failed to release file lock after retries: " + path.getAbsolutePath(), last);
    }

    /**
     * finally 블록에서 쓰기 위한 release wrapper입니다.
     *
     * <p>builder 코드에서는 lock을 잡은 뒤 예외가 나더라도 finally에서 반드시 풀어야 합니다.
     * 그런데 finally 안에서 checked exception을 다시 던지기 애매한 호출부가 많습니다.
     * 그렇다고 release 실패를 조용히 무시하면 stale lock이 남고, 다음 실행이 timeout으로
     * 막힐 수 있습니다. 그래서 이 wrapper는 {@link IOException}을 {@link IllegalStateException}
     * 으로 바꿔 올려서 실패가 테스트와 운영 로그에 드러나게 합니다.</p>
     */
    public void releaseUnchecked() {
        try {
            release();
        } catch (IOException e) {
            throw new IllegalStateException("failed to release file lock: " + path.getAbsolutePath(), e);
        }
    }

    public boolean isAcquired() {
        return acquired;
    }

    @Override
    public void close() throws IOException {
        release();
    }

    private boolean tryCreateOwnerFile() throws IOException {
        // lock 획득의 기준은 이 한 줄입니다. 파일이 없어서 새로 만들 수 있으면 성공,
        // 이미 있으면 실패입니다. 여러 프로세스가 동시에 호출해도 성공하는 쪽은 하나뿐이므로
        // 별도의 공유 메모리나 DB 없이도 "한 명만 들어가기" 규칙을 만들 수 있습니다.
        if (!path.createNewFile()) {
            return false;
        }
        boolean success = false;
        try (FileOutputStream out = new FileOutputStream(path, false)) {
            // 파일을 만든 직후 owner 정보를 씁니다. 이 정보는 lock을 잡는 데 필요한
            // 조건은 아니지만, timeout이 났을 때 누가 잡고 있는지 보여 주고 stale lock
            // 후보를 판단할 때 사용합니다.
            out.write(ownerText.getBytes(StandardCharsets.UTF_8));
            success = true;
            return true;
        } finally {
            if (!success) {
                // 드문 경우지만, lock 파일 생성에는 성공했는데 owner 정보 쓰기에 실패할 수 있습니다.
                // 이 상태로 남기면 token이 없어 정상 release도 stale 판단도 어렵습니다.
                // 따라서 아직 이 객체가 만든 직후일 때 바로 치웁니다.
                deleteCreatedLockFile();
            }
        }
    }

    private void deleteCreatedLockFile() throws IOException {
        // owner 정보를 쓰기 전에 문제가 난 lock 파일을 정리합니다. 이 경우에도 Windows에서
        // 파일 handle이 잠깐 남을 수 있으므로 일반 release와 같은 방식으로 재시도합니다.
        IOException last = null;
        for (int attempt = 0; attempt <= RELEASE_RETRY_DELAYS_MILLIS.length; attempt++) {
            if (!path.exists() || path.delete()) {
                return;
            }
            last = new IOException("failed to delete partially-created lock file: " + path.getAbsolutePath());
            if (attempt < RELEASE_RETRY_DELAYS_MILLIS.length) {
                sleepMillis(
                        RELEASE_RETRY_DELAYS_MILLIS[attempt],
                        "deleting partially-created lock file: " + path.getAbsolutePath());
            }
        }
        throw last;
    }

    private void releaseOnce() throws IOException {
        // 이미 파일이 없다면 release가 한 번 더 호출된 상황으로 보고 성공 처리합니다.
        // 중요한 것은 "내가 아닌 다른 사람의 lock을 지우지 않는 것"입니다.
        if (!path.exists()) {
            return;
        }
        String existing = readLockText(path);
        String existingToken = field(existing, "token");
        if (!token.equals(existingToken)) {
            // 같은 path라도 token이 다르면 다른 프로세스가 새로 만든 lock일 수 있습니다.
            // 예를 들어 오래 기다리던 프로세스가 timeout 직전에 release를 호출하는 사이,
            // 다른 프로세스가 같은 .lock 파일을 새로 만들었을 수 있습니다. 여기서 삭제하면
            // 그 프로세스의 보호막을 걷어 내는 셈이므로 즉시 실패합니다.
            throw new OwnerMismatchException(
                    "refusing to release file lock owned by another token: " + path.getAbsolutePath()
                            + " expected_token=" + token
                            + " existing_lock={" + summarize(existing) + "}");
        }
        if (!path.delete()) {
            throw new IOException("failed to delete file lock: " + path.getAbsolutePath());
        }
    }

    private boolean maybeDeleteStaleLock() {
        // stale lock은 "작업하던 프로세스가 죽어서 .lock 파일만 남은 상황"을 말합니다.
        // 이런 파일이 남으면 새 프로세스는 계속 기다리다가 timeout이 납니다. 다만 자동 삭제는
        // 위험하므로 기본값은 꺼져 있습니다. 운영자가 시간을 지정한 경우에만 아래 검사를 합니다.
        if (staleLockMillis <= 0L || !safeIsFile(path)) {
            return false;
        }
        long modified = safeLastModified(path);
        // lock 파일이 충분히 오래되지 않았다면 정상 작업 중일 가능성이 큽니다.
        // 이 단계에서는 본문도 읽지 않고 그대로 둡니다.
        if (modified <= 0L || System.currentTimeMillis() - modified < staleLockMillis) {
            return false;
        }
        try {
            String existing = readLockText(path);
            String existingToken = field(existing, "token");
            String existingHost = field(existing, "host");
            String existingPid = field(existing, "pid");
            if (existingToken == null || existingHost == null || existingPid == null) {
                return false;
            }
            // 공유 디렉터리를 여러 머신이 같이 볼 수 있습니다. 다른 host에서 만든 lock이면
            // 이 JVM이 그 pid의 생존 여부를 안전하게 확인할 수 없습니다. 이 경우에는 오래됐더라도
            // 절대 지우지 않습니다.
            if (!CURRENT_HOST.equals(existingHost)) {
                return false;
            }
            Boolean alive = isProcessAlive(existingPid);
            // alive == null은 "살았는지 죽었는지 모르겠다"는 뜻입니다.
            // 모를 때는 삭제하지 않습니다. 이 lock 클래스의 원칙은 "확실히 내 것이거나,
            // 확실히 죽은 프로세스의 것일 때만 삭제한다"입니다.
            if (alive == null || alive.booleanValue()) {
                return false;
            }

            // 삭제 직전에 token을 다시 읽습니다. 위에서 본문을 읽은 뒤 실제 삭제하기 전까지
            // 다른 프로세스가 lock을 새로 만들었을 수 있기 때문입니다. token이 바뀌었으면
            // 이제 더 이상 "방금 확인한 오래된 lock"이 아니므로 삭제하지 않습니다.
            String beforeDelete = readLockText(path);
            if (!existingToken.equals(field(beforeDelete, "token"))) {
                return false;
            }
            return deletePathWithRetries("deleting stale file lock: " + path.getAbsolutePath());
        } catch (IOException ignored) {
            // 판단 중 I/O 문제가 나면 lock을 지우지 않고 계속 기다립니다.
            // 최종 실패 원인은 acquire timeout에서 기존 lock 본문과 함께 보고합니다.
            return false;
        } catch (SecurityException ignored) {
            // 권한 문제도 판단 불가로 봅니다. stale 회수보다 오삭제 방지가 우선입니다.
            return false;
        }
    }

    private boolean deletePathWithRetries(String action) throws IOException {
        // Windows 현장에서 자주 보는 문제를 위한 방어입니다. 우리 프로세스는 파일을 닫았지만
        // 백신, IDE, 인덱서가 아주 짧게 파일을 열고 있으면 delete()가 실패할 수 있습니다.
        // 이 경우 바로 실패시키면 정상 작업도 불안정해지므로 짧은 간격으로 몇 번만 다시 시도합니다.
        IOException last = null;
        for (int attempt = 0; attempt <= RELEASE_RETRY_DELAYS_MILLIS.length; attempt++) {
            if (!path.exists() || path.delete()) {
                return true;
            }
            last = new IOException("failed while " + action);
            if (attempt < RELEASE_RETRY_DELAYS_MILLIS.length) {
                sleepMillis(RELEASE_RETRY_DELAYS_MILLIS[attempt], action);
            }
        }
        throw last;
    }

    private String describeExistingLock() {
        // acquire timeout 메시지에 기존 lock 내용을 붙이기 위한 helper입니다.
        // 장애 상황에서는 "누가 lock을 잡았는지"가 가장 중요한 정보라서 가능한 한 읽어 봅니다.
        try {
            if (!safeExists(path)) {
                return "missing";
            }
            return summarize(readLockText(path));
        } catch (IOException e) {
            return "unreadable path=" + path.getAbsolutePath() + " error=" + e.getMessage();
        } catch (SecurityException e) {
            return "unreadable path=" + path.getAbsolutePath() + " error=" + e.getMessage();
        }
    }

    private static String buildOwnerText(File path, String token) {
        // lock 파일은 사람이 메모장으로 열어 볼 수 있는 plain text로 둡니다.
        // token은 release 검증용이고, pid/thread/host/path는 timeout이나 현장 장애 분석용입니다.
        StringBuilder sb = new StringBuilder();
        sb.append("token=").append(token).append('\n');
        sb.append("pid=").append(CURRENT_PID).append('\n');
        sb.append("thread=").append(Thread.currentThread().getName()).append('\n');
        sb.append("host=").append(CURRENT_HOST).append('\n');
        sb.append("created_at_ms=").append(System.currentTimeMillis()).append('\n');
        sb.append("path=").append(path.getAbsolutePath()).append('\n');
        return sb.toString();
    }

    private static String readLockText(File file) throws IOException {
        return new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8);
    }

    private static String field(String text, String name) {
        String prefix = name + "=";
        String[] lines = text.split("\\r?\\n");
        for (String line : lines) {
            if (line.startsWith(prefix)) {
                return line.substring(prefix.length());
            }
        }
        return null;
    }

    private static String summarize(String text) {
        return text.replace('\r', ' ').replace('\n', ' ').trim();
    }

    private static long defaultStaleLockMillis() {
        // stale lock 자동 회수는 위험할 수 있으므로 기본값은 0, 즉 꺼짐입니다.
        // 필요한 환경에서만 JVM property나 환경 변수로 명시적으로 켭니다.
        String value = safeProperty(STALE_MILLIS_PROPERTY);
        if (value == null || value.trim().isEmpty()) {
            value = safeEnv(STALE_MILLIS_ENV);
        }
        if (value == null || value.trim().isEmpty()) {
            return 0L;
        }
        try {
            return Math.max(0L, Long.parseLong(value.trim()));
        } catch (NumberFormatException e) {
            // 설정값이 잘못되었을 때는 "마구 지우는" 방향보다 "안 지우는" 방향이 안전합니다.
            return 0L;
        }
    }

    private static String resolveCurrentPid() {
        try {
            String name = ManagementFactory.getRuntimeMXBean().getName();
            int at = name.indexOf('@');
            return at > 0 ? name.substring(0, at) : name;
        } catch (SecurityException e) {
            // pid를 못 읽으면 나중에 stale lock 회수에서 "프로세스가 죽었는지"를
            // 확인하기 어렵습니다. 그래도 일반 lock 획득/해제는 token만으로 동작하므로,
            // 여기서는 실패시키지 않고 빈 값으로 기록합니다.
            return "";
        }
    }

    private static String resolveCurrentHost() {
        try {
            return InetAddress.getLocalHost().getHostName();
        } catch (Exception e) {
            // host를 못 읽어도 일반 lock 동작은 가능합니다. 다만 stale 회수에서는
            // 다른 host lock을 지우지 않는 쪽으로 보수적으로 동작하게 됩니다.
            return "unknown";
        }
    }

    private static Boolean isProcessAlive(String pidText) {
        // stale lock 회수에서만 쓰는 helper입니다.
        // 반환값은 일부러 boolean이 아니라 세 가지 상태를 표현합니다.
        // TRUE  = 해당 pid가 살아 있다고 판단했습니다.
        // FALSE = 해당 pid가 죽었다고 판단했습니다.
        // NULL  = 권한, 플랫폼, 실행 실패 때문에 판단할 수 없습니다. 이 경우 lock 삭제 금지입니다.
        Long pid = parseLong(pidText);
        if (pid == null || pid.longValue() <= 0L) {
            return null;
        }
        if (String.valueOf(pid.longValue()).equals(CURRENT_PID)) {
            // 현재 JVM의 pid라면 당연히 살아 있습니다. 자기 lock을 stale로 오해하지 않게 합니다.
            return Boolean.TRUE;
        }
        if (isWindows()) {
            return isWindowsProcessAlive(pid.longValue());
        }
        try {
            File proc = new File("/proc/" + pid.longValue());
            if (safeIsDirectory(new File("/proc"))) {
                // Linux 계열에서는 보통 /proc/<pid>가 있으면 프로세스가 살아 있다고 볼 수 있습니다.
                // 단, 보안 정책 때문에 접근이 막힌 경우를 "죽음"으로 착각하면 안 됩니다.
                // 그래서 실패 시 false가 아니라 null을 반환할 수 있는 helper를 씁니다.
                return safeExistsOrUnknown(proc);
            }
        } catch (SecurityException e) {
            return null;
        }
        return null;
    }

    private static Boolean isWindowsProcessAlive(long pid) {
        // Java 8 대상 코드라 ProcessHandle을 사용할 수 없습니다. Windows에서는 tasklist로
        // pid가 보이는지 확인합니다. 단, tasklist 실행 실패, timeout, interrupt는 모두
        // "판단 불가"로 처리합니다. 프로세스 확인이 불안정할 때 lock을 지우는 것은 위험합니다.
        Process process = null;
        try {
            process = new ProcessBuilder("tasklist", "/FI", "PID eq " + pid, "/NH").redirectErrorStream(true).start();
            boolean completed = process.waitFor(2L, TimeUnit.SECONDS);
            if (!completed) {
                process.destroy();
                return null;
            }
            String output = new String(readAllBytes(process), StandardCharsets.UTF_8);
            return Boolean.valueOf(output.contains(String.valueOf(pid)));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        } catch (Exception e) {
            return null;
        } finally {
            if (process != null) {
                process.destroy();
            }
        }
    }

    private static byte[] readAllBytes(Process process) throws IOException {
        byte[] buffer = new byte[4096];
        int read;
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        while ((read = process.getInputStream().read(buffer)) >= 0) {
            out.write(buffer, 0, read);
        }
        return out.toByteArray();
    }

    private static Long parseLong(String text) {
        if (text == null) {
            return null;
        }
        try {
            return Long.valueOf(text.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static boolean isWindows() {
        // "Darwin" 같은 문자열에도 "win"이 들어가므로 contains("win")은 쓰지 않습니다.
        // os.name을 읽지 못하면 Windows 전용 tasklist 확인을 하지 않고 "판단 불가" 쪽으로 갑니다.
        String os = safeProperty("os.name");
        return os != null && os.toLowerCase(java.util.Locale.ROOT).startsWith("windows");
    }

    private static String safeProperty(String name) {
        // 일부 제한된 실행 환경에서는 system property 접근이 막힐 수 있습니다.
        // lock 기능 자체가 죽지 않도록 읽기 실패를 null로 접습니다.
        try {
            return System.getProperty(name);
        } catch (SecurityException e) {
            return null;
        }
    }

    private static String safeEnv(String name) {
        // 환경 변수 접근도 보안 정책에 막힐 수 있으므로 property와 같은 방식으로 처리합니다.
        try {
            return System.getenv(name);
        } catch (SecurityException e) {
            return null;
        }
    }

    private static boolean safeExists(File file) {
        // 진단 메시지용 존재 확인입니다. 못 읽으면 "없는 것처럼" 다루되,
        // stale lock 삭제 판단에는 이 helper를 쓰지 않습니다.
        try {
            return file.exists();
        } catch (SecurityException e) {
            return false;
        }
    }

    private static Boolean safeExistsOrUnknown(File file) {
        // stale lock 판단에서는 권한 문제를 "없다"로 보면 위험합니다.
        // 그래서 SecurityException은 false가 아니라 null, 즉 판단 불가로 반환합니다.
        try {
            return Boolean.valueOf(file.exists());
        } catch (SecurityException e) {
            return null;
        }
    }

    private static boolean safeIsFile(File file) {
        // 이 helper는 stale 후보 진입 전에만 씁니다. 권한 문제면 후보가 아니라고 보고
        // 삭제를 시도하지 않습니다.
        try {
            return file.isFile();
        } catch (SecurityException e) {
            return false;
        }
    }

    private static boolean safeIsDirectory(File file) {
        // /proc 사용 가능 여부 확인용입니다. 권한 문제면 /proc 기반 판단을 포기합니다.
        try {
            return file.isDirectory();
        } catch (SecurityException e) {
            return false;
        }
    }

    private static long safeLastModified(File file) {
        // 파일 수정 시간을 읽지 못하면 오래된 lock인지 판단할 수 없습니다.
        // 0을 반환하면 maybeDeleteStaleLock()이 삭제를 포기합니다.
        try {
            return file.lastModified();
        } catch (SecurityException e) {
            return 0L;
        }
    }

    private static void sleepMillis(long millis, String action) throws IOException {
        // lock 획득/해제 대기 중 interrupt가 들어오면 interrupt 상태를 복구하고
        // IOException으로 올립니다. 호출자는 lock 작업 실패로 처리할 수 있습니다.
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted while " + action, e);
        }
    }

    private static final class OwnerMismatchException extends IOException {
        // release 중 token이 달라졌다는 것은 "이 파일은 이제 내 lock이 아닐 수 있다"는 뜻입니다.
        // 일반 delete 실패보다 더 위험하므로 별도 타입으로 즉시 중단시킵니다.
        OwnerMismatchException(String message) {
            super(message);
        }
    }
}
